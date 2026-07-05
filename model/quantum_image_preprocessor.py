"""
model/quantum_image_preprocessor.py

Quantum Hadamard Edge Detection (QHED) preprocessing on raw images.

Applied BEFORE the ViT backbone — the only quantum component in an otherwise
classical pipeline. Tests whether quantum edge preprocessing improves re-ID.

Pipeline per frame:
    [N, C, H, W]
    → bilinear downsample to [N, C, 32, 32]
    → unfold into 8×8 patches → [N*C*16, 64]
    → normalize (AmplitudeEmbedding requires ||x||=1)
    → 6-qubit circuit:
          AmplitudeEmbedding     — encode patch as quantum state
          Hadamard on all qubits — QHED: rotate to edge-sensitive basis
          StronglyEntanglingLayers (trainable) — learn which edges matter
          measure probs          → [N*C*16, 64]
    → reshape → [N, C, 32, 32]
    → bilinear upsample → [N, C, H, W]
    → x_out = x + alpha * edge_map   (learnable residual, alpha init 0.01)

The ViT backbone sees the original image plus a small quantum-derived edge signal.
All CLIP backbone weights are unchanged. Only qlayer_weights and alpha are trained.

alpha near 0 at init → output ≈ original image → stable early training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuantumImagePreprocessor(nn.Module):
    """
    QHED preprocessing module. Insert before image_encoder in forward pass.

    Args:
        n_layers    (int): StronglyEntanglingLayers after Hadamard. 0 = pure QHED.
        patch_size  (int): Patch side length. Default 8 (64 pixels = 6 qubits).
        small_size  (int): Downsample target. Default 32 (→ 16 patches per channel).
        device_name (str): PennyLane device.
    """

    PATCH_PIX = 64   # patch_size^2 = 8^2
    N_QUBITS  = 6    # 2^6 = 64

    def __init__(self, n_layers: int = 1, patch_size: int = 8,
                 small_size: int = 32, device_name: str = 'default.qubit'):
        super().__init__()
        self.n_layers   = n_layers
        self.patch_size = patch_size
        self.small_size = small_size
        self.n_q        = self.N_QUBITS
        self.n_patches  = (small_size // patch_size) ** 2  # 16 for 32/8

        dev = qml.device(device_name, wires=self.n_q)

        if n_layers > 0:
            weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, self.n_q)
            self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.01)

            @qml.qnode(dev, interface='torch', diff_method='backprop')
            def _circuit(amplitudes, weights):
                # amplitudes: [N, 64] — one normalised patch per sample
                # weights:    [n_layers, n_q, 3] — trainable
                qml.AmplitudeEmbedding(amplitudes, wires=range(self.n_q), normalize=False)
                # QHED: Hadamard on all qubits rotates to edge-sensitive basis
                for i in range(self.n_q):
                    qml.Hadamard(wires=i)
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_q))
                return qml.probs(wires=range(self.n_q))
        else:
            @qml.qnode(dev, interface='torch', diff_method='backprop')
            def _circuit(amplitudes, weights):
                qml.AmplitudeEmbedding(amplitudes, wires=range(self.n_q), normalize=False)
                for i in range(self.n_q):
                    qml.Hadamard(wires=i)
                return qml.probs(wires=range(self.n_q))

        self.circuit = _circuit

        # Learnable mixing weight — init at 1.0 for full quantum signal from the start
        self.alpha = nn.Parameter(torch.tensor(1.0))

        print(f"[QuantumImagePreprocessor] n_qubits={self.n_q}, patch={patch_size}×{patch_size}, "
              f"n_patches={self.n_patches}/channel, n_layers={n_layers}")

    def _apply(self, fn):
        super()._apply(fn)
        if self.n_layers > 0:
            self.qlayer_weights.data = self.qlayer_weights.data.cpu().float()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C, H, W] — normalised images (float32 or float16)
        returns: [N, C, H, W] — x + alpha * quantum_edge_map
        """
        N, C, H, W = x.shape
        p = self.patch_size
        s = self.small_size
        n_p = s // p  # patches per spatial dim (4 for 32/8)

        # Downsample to 32×32
        x_small = F.interpolate(x.float(), size=(s, s),
                                mode='bilinear', align_corners=False)  # [N, C, 32, 32]

        # Extract non-overlapping patches: [N, C, n_p, n_p, p, p]
        patches = x_small.unfold(2, p, p).unfold(3, p, p)            # [N, C, n_p, n_p, p, p]
        patches = patches.contiguous().view(N * C * n_p * n_p, self.PATCH_PIX)  # [N*C*16, 64]

        # Normalise for AmplitudeEmbedding (||x||=1 required)
        norms   = patches.norm(dim=1, keepdim=True).clamp(min=1e-8)
        patches_norm = (patches / norms).cpu().float()                # [N*C*16, 64]

        # Run QHED circuit — single batched call over all patches
        if self.n_layers > 0:
            q_out = self.circuit(patches_norm,
                                 self.qlayer_weights.cpu().float()).float()
        else:
            q_out = self.circuit(patches_norm,
                                 torch.zeros(1)).float()               # dummy weights

        # q_out: [N*C*16, 64] — measurement probability over 64 basis states

        # Reconstruct spatial structure: fold patches back to [N, C, 32, 32]
        q_edge = q_out.view(N, C, n_p, n_p, p, p)
        q_edge = q_edge.permute(0, 1, 2, 4, 3, 5).contiguous()
        q_edge = q_edge.view(N, C, s, s)                              # [N, C, 32, 32]

        # Upsample edge map to original image size
        q_edge = F.interpolate(q_edge.to(x.device), size=(H, W),
                               mode='bilinear', align_corners=False)  # [N, C, H, W]

        # Residual addition: alpha near 0 → ViT sees mostly original image at init
        return (x.float() + self.alpha * q_edge).to(x.dtype)

    def extra_repr(self):
        return (f"n_qubits={self.n_q}, patch_size={self.patch_size}, "
                f"small_size={self.small_size}, n_patches={self.n_patches}/channel, "
                f"n_layers={self.n_layers}")
