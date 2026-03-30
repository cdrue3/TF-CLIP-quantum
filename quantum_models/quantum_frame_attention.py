"""
quantum_models/quantum_frame_attention.py

Quantum Frame Attention — VQC generates soft attention weights over T frames.

Instead of replacing temporal aggregation (TQA, failed), the VQC produces T scalar
weights to do a soft weighted sum of frame features. The multi-class bottleneck is
completely avoided — the VQC outputs only T scalars (e.g. 4), not 625 logits.

Architecture:
    [B*T, 768] frame features (from backbone, before temporal mean)
      → reshape [B, T, 768]
      → per-frame compress: Linear(768, n_qubits) shared   [B, T, n_qubits]
      → sigmoid * π                                         [B, T, n_qubits]
      → VQC: T parallel evals per sample → [B, T, 2^n_qubits]
      → weight_net: Linear(2^n_qubits, 1) → squeeze → [B, T]
      → softmax over T → attention weights [B, T] ∑=1
      → weighted sum: Σ_t w_t * frame_t → [B, 768]
      → replaces the plain .mean(1) temporal aggregation in the backbone

bypass_quantum=True: replace VQC with Linear(n_qubits, 2^n_qubits)+ReLU.
Classical ablation tests whether *learned* temporal attention (not quantum)
explains any gains, separating the architecture from the quantum component.

Integration note:
    This module is called INSIDE the forward pass, between per-frame feature
    extraction and the temporal mean. The standard TF-CLIP path is:
        img_feature = image_features[:, 0].view(B, T, -1).mean(1)   # [B, 768]
    With QuantumFrameAttention:
        img_feature = self.frame_attn(image_features[:, 0].view(B, T, -1))  # [B, 768]
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumFrameAttention(nn.Module):
    """
    VQC-based soft attention over T video frames.

    Given [B, T, 768] frame features, outputs [B, 768] weighted sum.

    Args:
        in_features  (int): Frame feature dimension (768 for ViT-B-16).
        n_qubits     (int): Number of qubits. Default 8 → 256 quantum features.
        n_layers     (int): StronglyEntanglingLayers depth. Default 2.
        bypass_quantum (bool): Replace VQC with Linear+ReLU for ablation.
        device_name  (str): PennyLane device. Default 'default.qubit'.
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_quantum_features = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        # Shared per-frame compression: 768 → n_qubits
        self.pre_net = nn.Linear(in_features, n_qubits, bias=False)

        if bypass_quantum:
            self.classical_expansion = nn.Sequential(
                nn.Linear(n_qubits, self.n_quantum_features, bias=False),
                nn.ReLU(),
            )
        else:
            dev = qml.device(device_name, wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return qml.probs(wires=range(n_qubits))

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            self.qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

        # Map quantum features → scalar attention logit per frame
        self.weight_net = nn.Linear(self.n_quantum_features, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        if self.bypass_quantum:
            nn.init.kaiming_normal_(self.classical_expansion[0].weight, a=0, mode="fan_in")
        else:
            nn.init.normal_(self.qlayer.weights, mean=0, std=0.01)
        # weight_net: near-zero weights, zero bias → uniform attention at init
        nn.init.normal_(self.weight_net.weight, mean=0, std=0.001)
        nn.init.zeros_(self.weight_net.bias)

    def to(self, *args, **kwargs):
        """Pin qlayer to CPU."""
        super().to(*args, **kwargs)
        if not self.bypass_quantum:
            self.qlayer.to(device=torch.device("cpu"), dtype=torch.float32)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, in_features]  — per-frame features before temporal mean
        Returns:
            out: [B, in_features]   — attention-weighted temporal aggregation
        """
        B, T, D = x.shape
        input_dtype  = x.dtype
        input_device = x.device
        x_f = x.float()

        # Compress each frame: [B, T, D] → [B, T, n_qubits]
        x_flat = x_f.view(B * T, D)
        angles = torch.sigmoid(self.pre_net(x_flat)) * math.pi   # [B*T, n_qubits]

        if self.bypass_quantum:
            q_feat = self.classical_expansion(angles)              # [B*T, 2^n_qubits]
        else:
            angles_cpu = angles.cpu().float()
            q_feat = self.qlayer(angles_cpu).to(input_device)     # [B*T, 2^n_qubits]

        q_feat = q_feat.view(B, T, self.n_quantum_features)       # [B, T, 2^n_qubits]

        # Per-frame scalar attention logit → softmax over T frames
        attn_logits = self.weight_net(q_feat).squeeze(-1)         # [B, T]
        attn_weights = torch.softmax(attn_logits, dim=-1)         # [B, T]

        # Weighted sum over frames
        out = (attn_weights.unsqueeze(-1) * x_f).sum(dim=1)      # [B, D]

        return out.to(input_dtype)

    def extra_repr(self) -> str:
        mode = "classical_bypass" if self.bypass_quantum else "VQC"
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, mode={mode}"
        )
