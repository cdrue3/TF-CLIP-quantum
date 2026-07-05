"""
quantum_models/quantum_interlaced.py

Q-C-Q Interlaced Adapter — two VQC stages with a classical bottleneck between them.

Survey paper's best reported pattern (brain tumor +5.7%). The interlaced design
avoids the single-pass dimensionality bottleneck by running:
  quantum → classical → quantum

Architecture:
    x [B, in_features=768]
      Stage 1: pre_net1(768→n_q) → VQC1 → probs [2^n_q] → upscale1(2^n_q→256) → ReLU
      Classical: Linear(256→256) → ReLU                   [B, 256]
      Stage 2: pre_net2(256→n_q) → VQC2 → probs [2^n_q] → upscale2(2^n_q→768)   [B, 768]
      Residual: x + stage2_output                          [B, 768]

The first VQC processes raw CLIP features; the classical middle distils them into a
256-dim latent; the second VQC processes the latent; the final upscale produces the
768-dim residual delta. The residual connection protects against gradient collapse.

bypass_quantum=True: both VQCs → Linear(n_q→2^n_q)+ReLU.
Ablation: is it the quantum circuits or just the Q-C-Q architecture (two-stage residual)
that matters?

Init strategy:
  upscale2 std=0.001 → residual delta ≈ 0 at init (identity start, same as adapter).
  upscale1/classical/pre_net: kaiming_normal (standard).
  VQC: std=0.01 (near-identity, avoids barren plateau at init).
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumInterlacedAdapter(nn.Module):
    """
    Q-C-Q interlaced residual adapter.

    x_out = x + upscale2(VQC2(pre_net2(classical(upscale1(VQC1(pre_net1(x)))))))

    Args:
        in_features  (int): Feature dimension (768 for ViT-B-16).
        n_qubits     (int): Number of qubits (both VQCs). Default 8.
        n_layers     (int): StronglyEntanglingLayers depth (both VQCs). Default 2.
        mid_features (int): Classical bottleneck dimension. Default 256.
        bypass_quantum (bool): Replace both VQCs with Linear+ReLU for ablation.
        device_name  (str): PennyLane device. Default 'default.qubit'.
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        mid_features: int = 256,
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_quantum_features = 2 ** n_qubits
        self.mid_features = mid_features
        self.bypass_quantum = bypass_quantum

        # --- Stage 1: in_features → n_qubits → 2^n_qubits → mid_features ---
        self.pre_net1 = nn.Linear(in_features, n_qubits, bias=False)

        if bypass_quantum:
            self.classical_q1 = nn.Sequential(
                nn.Linear(n_qubits, self.n_quantum_features, bias=False),
                nn.ReLU(),
            )
        else:
            dev1 = qml.device(device_name, wires=n_qubits)

            @qml.qnode(dev1, interface="torch", diff_method="backprop")
            def _circuit1(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return qml.probs(wires=range(n_qubits))

            weight_shapes1 = {"weights": (n_layers, n_qubits, 3)}
            self.qlayer1 = qml.qnn.TorchLayer(_circuit1, weight_shapes1)

        self.upscale1 = nn.Linear(self.n_quantum_features, mid_features, bias=False)

        # --- Classical bottleneck: mid_features → mid_features ---
        self.classical_mid = nn.Sequential(
            nn.Linear(mid_features, mid_features, bias=False),
            nn.ReLU(),
        )

        # --- Stage 2: mid_features → n_qubits → 2^n_qubits → in_features ---
        self.pre_net2 = nn.Linear(mid_features, n_qubits, bias=False)

        if bypass_quantum:
            self.classical_q2 = nn.Sequential(
                nn.Linear(n_qubits, self.n_quantum_features, bias=False),
                nn.ReLU(),
            )
        else:
            dev2 = qml.device(device_name, wires=n_qubits)

            @qml.qnode(dev2, interface="torch", diff_method="backprop")
            def _circuit2(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return qml.probs(wires=range(n_qubits))

            weight_shapes2 = {"weights": (n_layers, n_qubits, 3)}
            self.qlayer2 = qml.qnn.TorchLayer(_circuit2, weight_shapes2)

        self.upscale2 = nn.Linear(self.n_quantum_features, in_features, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.pre_net1.weight, a=0, mode="fan_in")
        nn.init.kaiming_normal_(self.pre_net2.weight, a=0, mode="fan_in")
        nn.init.kaiming_normal_(self.upscale1.weight, a=0, mode="fan_in")
        nn.init.kaiming_normal_(self.classical_mid[0].weight, a=0, mode="fan_in")
        if self.bypass_quantum:
            nn.init.kaiming_normal_(self.classical_q1[0].weight, a=0, mode="fan_in")
            nn.init.kaiming_normal_(self.classical_q2[0].weight, a=0, mode="fan_in")
        else:
            nn.init.normal_(self.qlayer1.weights, mean=0, std=0.01)
            nn.init.normal_(self.qlayer2.weights, mean=0, std=0.01)
        # upscale2: near-zero so residual delta ≈ 0 at init
        nn.init.normal_(self.upscale2.weight, mean=0, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]
        Returns:
            x_out: [B, in_features] (same shape/dtype/device as input)
        """
        input_dtype  = x.dtype
        x_f = x.float()

        # Stage 1: in_features → mid_features via VQC1
        angles1 = torch.sigmoid(self.pre_net1(x_f)) * math.pi   # [B, n_qubits]
        if self.bypass_quantum:
            q1_feat = self.classical_q1(angles1)
        else:
            q1_feat = self.qlayer1(angles1.float())
        mid = torch.relu(self.upscale1(q1_feat))                 # [B, mid_features]

        # Classical bottleneck
        mid = self.classical_mid(mid)                             # [B, mid_features]

        # Stage 2: mid_features → in_features via VQC2
        angles2 = torch.sigmoid(self.pre_net2(mid)) * math.pi   # [B, n_qubits]
        if self.bypass_quantum:
            q2_feat = self.classical_q2(angles2)
        else:
            q2_feat = self.qlayer2(angles2.float())
        delta = self.upscale2(q2_feat)                           # [B, in_features]

        # Residual
        x_out = x_f + delta

        return x_out.to(input_dtype)

    def extra_repr(self) -> str:
        mode = "classical_bypass" if self.bypass_quantum else "VQC"
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"mid_features={self.mid_features}, n_layers={self.n_layers}, mode={mode}"
        )
