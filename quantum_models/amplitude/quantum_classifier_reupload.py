"""
quantum_models/amplitude/quantum_classifier_reupload.py

QClassifier with block-wise data re-uploading — no feature compression.

Architecture:
    Input x: [B, T, in_features]
    1. Reshape to [B*T, in_features]  (process each frame independently)
    2. Split in_features into n_blocks = in_features // n_qubits blocks of n_qubits values
    3. Re-uploading circuit (shared weights across all blocks AND frames):
           for block in range(n_blocks):
               AngleEmbedding(x[block*n_q:(block+1)*n_q], wires, rotation='Y')
               StronglyEntanglingLayers(shared_weights, wires)
       qml.probs(wires) → [2^n_qubits]
    4. Reshape [B*T, 2^n_q] → [B, T, 2^n_q] → mean(1) → [B, 2^n_q]
    5. Upscale: Linear(2^n_q, in_features)
    6. Skip: output = mean_pool(x) + delta

No pre_net compression — all in_features encoded via 96 re-uploading blocks (for
n_qubits=8, in_features=768). Shared weights across blocks: 36 params total.

Key difference from QTD/QTemporal: re-uploading is over FEATURE blocks, not time.
Temporal pooling is classical mean-pool AFTER the circuit outputs are averaged.

Optional preprocessing (quantum-inspired, applied to [B*T, in_features] before encoding):
  none  — raw features
  edge  — replace each frame with diff to next frame (highlights change)
  phase — sine-encode features before angle embedding (Fourier feature map)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QClassifierReupload(nn.Module):
    """
    Block-wise re-uploading quantum classifier. No 768→n_qubits compression.

    Args:
        in_features  (int): Feature dimension (768 for ViT-B/16).
        n_qubits     (int): Qubit count. Determines block size. Default 8.
        n_layers     (int): StronglyEntanglingLayers depth per block. Default 2.
        seq_len      (int): T — frames per tracklet. Default 8.
        preprocess   (str): 'none' | 'edge' | 'phase'. Applied before encoding.
        bypass_quantum (bool): Return plain mean-pool if True.
    """

    def __init__(
        self,
        in_features: int = 768,
        n_qubits: int = 8,
        n_layers: int = 2,
        seq_len: int = 8,
        preprocess: str = 'none',
        bypass_quantum: bool = False,
        device_name: str = 'default.qubit',
    ):
        super().__init__()
        assert in_features % n_qubits == 0, (
            f"in_features ({in_features}) must be divisible by n_qubits ({n_qubits})")

        self.in_features    = in_features
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.seq_len        = seq_len
        self.preprocess     = preprocess
        self.bypass_quantum = bypass_quantum
        self.n_blocks       = in_features // n_qubits   # 96 for 768/8
        self.n_measurements = 2 ** n_qubits             # 256

        if not bypass_quantum:
            n_q      = n_qubits
            n_blocks = self.n_blocks
            dev      = qml.device(device_name, wires=n_q)

            @qml.qnode(dev, interface='torch', diff_method='backprop')
            def _circuit(angles_2d, weights):
                # angles_2d: [n_blocks, B, n_q] — one block per re-upload step
                # weights:   [n_layers, n_q, 3] — shared across all blocks
                for b in range(n_blocks):
                    qml.AngleEmbedding(angles_2d[b], wires=range(n_q), rotation='Y')
                    qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                return qml.probs(wires=range(n_q))

            self.circuit = _circuit
            weight_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=n_layers, n_wires=n_q
            )
            self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))

        # Upscale 2^n_q → in_features; near-zero init so skip starts as mean_pool
        self.upscale = nn.Linear(self.n_measurements, in_features, bias=False)
        self._init_weights()

    def _init_weights(self):
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.01)
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D] (N = B*T flattened) → [N, D]
        Applies quantum-inspired feature transformation before angle encoding.
        """
        if self.preprocess == 'none':
            return x

        elif self.preprocess == 'edge':
            # Reshape to [B, T, D], compute frame diffs, flatten back
            # For frames where diff isn't available (last), use zero diff
            B = x.shape[0] // self.seq_len
            frames = x.view(B, self.seq_len, -1)            # [B, T, D]
            diffs  = frames[:, 1:] - frames[:, :-1]         # [B, T-1, D]
            # Pad last frame diff with zeros
            diffs  = F.pad(diffs, (0, 0, 0, 1))             # [B, T, D]
            return (frames + diffs).reshape(x.shape[0], -1) # [N, D]

        elif self.preprocess == 'phase':
            # Sine-based Fourier feature map — encode features as phase angles
            # Expands the expressiveness of angle encoding by pre-applying a
            # non-linear transformation that emphasises oscillatory structure
            return torch.sin(x * math.pi)

        else:
            raise ValueError(f"Unknown preprocess mode: {self.preprocess}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, in_features] → [B, in_features]
        """
        mean_feat = x.mean(1)  # [B, D] — skip connection

        if self.bypass_quantum:
            return mean_feat

        input_dtype  = x.dtype
        B, T, D = x.shape

        # Flatten to [B*T, D], apply preprocessing
        x_flat = x.float().reshape(B * T, D)                         # [N, D]
        x_flat = self._preprocess(x_flat)                            # [N, D]

        # Normalise to angle range (0, π) via sigmoid
        angles = torch.sigmoid(x_flat) * math.pi                     # [N, D]

        # Reshape to [n_blocks, N, n_qubits] for PennyLane broadcasting
        angles = angles.reshape(B * T, self.n_blocks, self.n_qubits) # [N, n_b, n_q]
        angles_in = angles.permute(1, 0, 2).float()                   # [n_b, N, n_q]

        # Single batched circuit call — PennyLane broadcasts over N
        q_out = self.circuit(
            angles_in,
            self.qlayer_weights.float()
        ).float()                                                     # [N, 2^n_q]

        # Pool over frames: [B*T, 2^n_q] → [B, T, 2^n_q] → mean → [B, 2^n_q]
        q_out = q_out.reshape(B, T, self.n_measurements).mean(1)

        # Upscale + skip
        delta = self.upscale(q_out)                                  # [B, D]
        return (mean_feat.float() + delta).to(dtype=input_dtype)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_blocks={self.n_blocks}, n_layers={self.n_layers}, "
            f"seq_len={self.seq_len}, preprocess={self.preprocess}, "
            f"n_measurements={self.n_measurements}, bypass={self.bypass_quantum}"
        )
