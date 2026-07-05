"""
quantum_models/quantum_channel_attention.py

Quantum Channel Attention — feature-level integration pattern.

Instead of classifying (sequential) or concatenating (parallel), the VQC generates
per-channel attention weights that multiplicatively reweight CLIP features:

    x [B, in_features]
      → pre_net:  Linear(in_features, n_qubits)     # compress
      → sigmoid(·) * π                               # encode angles ∈ (0, π)
      → VQC:      AngleEmbedding + StronglyEntangling → probs [B, 2^n_qubits]
      → expand:   Linear(2^n_qubits, in_features)    # per-channel weights
      → sigmoid                                       # ∈ (0, 1) — attention
      → x_out = x * weights + x                      # residual-gated attention

Key differences from QuantumAdapter (quantum_adapter.py):
  - Adapter:   x + upscale(VQC(x))          — additive residual
  - Channel:   x * sigmoid(expand(VQC(x))) + x  — multiplicative attention + residual

The multiplicative path means the quantum module cannot corrupt features it knows
nothing about — weights near 1.0 preserve the original feature; weights near 0 suppress.
The additive residual (+x) ensures the module starts near 2x (expand init near 0 means
sigmoid(0)=0.5, so x * 0.5 + x = 1.5x — slight scale up at init).

To make init truly neutral: expand weights near a constant c s.t. sigmoid(c)≈1.
We use expand bias=+4 → sigmoid(4)≈0.982 ≈ 1 — effectively identity at init.

bypass_quantum=True: replaces VQC with Linear(n_qubits, 2^n_qubits)+ReLU.
Ablation: does any benefit come from the quantum circuit specifically, or just from
the channel attention architecture?
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumChannelAttention(nn.Module):
    """
    Residual multiplicative channel attention via VQC.

    x_out = x * sigmoid(expand(VQC(pre_net(x)))) + x

    Args:
        in_features  (int): Feature dimension (e.g. 768 for ViT-B-16).
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

        # Pre-projection: in_features → n_qubits angles
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

        # Expand: 2^n_qubits → in_features per-channel attention logits.
        # bias=+4 → sigmoid(4) ≈ 0.982 ≈ 1 at init → attention ≈ identity.
        self.expand = nn.Linear(self.n_quantum_features, in_features, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        if self.bypass_quantum:
            nn.init.kaiming_normal_(self.classical_expansion[0].weight, a=0, mode="fan_in")
        else:
            nn.init.normal_(self.qlayer.weights, mean=0, std=0.01)
        # expand: near-zero weights + large positive bias → sigmoid ≈ 1 at init
        nn.init.normal_(self.expand.weight, mean=0, std=0.001)
        nn.init.constant_(self.expand.bias, 4.0)

    def to(self, *args, **kwargs):
        """Pin qlayer to CPU; all other modules follow device."""
        super().to(*args, **kwargs)
        if not self.bypass_quantum:
            self.qlayer.to(device=torch.device("cpu"), dtype=torch.float32)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]
        Returns:
            x_out: [B, in_features]  (same shape/dtype/device as input)
        """
        input_dtype  = x.dtype
        input_device = x.device
        x_f = x.float()

        angles = torch.sigmoid(self.pre_net(x_f)) * math.pi  # [B, n_qubits]

        if self.bypass_quantum:
            q_feat = self.classical_expansion(angles)          # [B, 2^n_qubits]
        else:
            angles_cpu = angles.float()
            q_feat = self.qlayer(angles_cpu).to(input_device)  # [B, 2^n_qubits]

        attn_weights = torch.sigmoid(self.expand(q_feat))      # [B, in_features] ∈ (0,1)

        # Multiplicative attention + residual: preserves original features at init
        x_out = x_f * attn_weights + x_f

        return x_out.to(input_dtype)

    def extra_repr(self) -> str:
        mode = "classical_bypass" if self.bypass_quantum else "VQC"
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, mode={mode}"
        )
