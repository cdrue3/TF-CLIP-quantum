"""
quantum_models/classification/quantum_multihead_classifier.py

Multi-Head VQC Classifier.

Addresses the single-head QClassifier bottleneck: a single 2^8=256-dim
VQC output is too limited for 1604 classes. Multi-head uses K independent
VQC heads, each producing 2^n_qubits features, concatenated to K*256-dim,
then projected to num_classes.

Architecture:
    x [B, in_features]
    → K independent heads, each:
        pre_net_k: Linear(in_features → n_qubits)
        sigmoid·π → VQC_k → probs [B, 2^n_q]
    → concat: [B, K * 2^n_q]
    → fusion_net: Linear(K*2^n_q → num_classes)

Total VQC params: K × n_layers × n_qubits × 3
  K=4, n_layers=2, n_qubits=8: 4×2×8×3 = 192 (vs 48 for single-head)

bypass_quantum=True: each head → Linear(in_features, 2^n_q)+ReLU; same shape.

Each head has independent pre_net + VQC weights — diverse projections
of the same feature space, analogous to multi-head attention.
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class _SingleHead(nn.Module):
    """Single VQC head: in_features → probs [2^n_qubits]."""

    def __init__(self, in_features, n_qubits, n_layers, bypass_quantum, device_name):
        super().__init__()
        self.n_qubits       = n_qubits
        self.n_measurements = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        self.pre_net = nn.Linear(in_features, n_qubits, bias=False)

        if bypass_quantum:
            self.bypass_net = nn.Sequential(
                nn.Linear(in_features, self.n_measurements),
                nn.ReLU(),
            )
        else:
            n_q = n_qubits
            dev = qml.device(device_name, wires=n_q)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _circuit(angles, weights):
                qml.AngleEmbedding(angles, wires=range(n_q), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                return qml.probs(wires=range(n_q))

            self.circuit = _circuit
            weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_q)
            self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))

        self._init_weights(in_features)

    def _init_weights(self, in_features):
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.01)


    def forward(self, x):
        if self.bypass_quantum:
            return self.bypass_net(x.float())

        input_device = x.device
        angles = torch.sigmoid(self.pre_net(x.float())) * math.pi  # [B, n_q]
        probs = self.circuit(
            angles.cpu().float(),
            self.qlayer_weights.float()
        ).float()  # [B, 2^n_q]
        return probs


class QuantumMultiHeadClassifier(nn.Module):
    """
    K independent VQC heads, concatenated and projected to num_classes.

    Args:
        in_features   (int): Input feature dimension. Default 768.
        num_classes   (int): Number of output classes.
        n_heads       (int): Number of independent VQC heads. Default 4.
        n_qubits      (int): Qubit count per head. Default 8.
        n_layers      (int): VQC depth per head. Default 2.
        bypass_quantum(bool): If True, each head uses Linear+ReLU instead of VQC.
        device_name   (str): PennyLane device.
    """

    def __init__(
        self,
        in_features: int = 768,
        num_classes: int = 1604,
        n_heads: int = 4,
        n_qubits: int = 8,
        n_layers: int = 2,
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features    = in_features
        self.num_classes    = num_classes
        self.n_heads        = n_heads
        self.n_qubits       = n_qubits
        self.n_measurements = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        # K independent VQC heads
        self.heads = nn.ModuleList([
            _SingleHead(in_features, n_qubits, n_layers, bypass_quantum, device_name)
            for _ in range(n_heads)
        ])

        # Fusion: K*2^n_q → num_classes
        self.fusion_net = nn.Linear(n_heads * self.n_measurements, num_classes)
        nn.init.normal_(self.fusion_net.weight, std=0.001)
        nn.init.zeros_(self.fusion_net.bias)

        total_vqc = n_heads * n_layers * n_qubits * 3
        print(
            f"[QuantumMultiHeadClassifier] n_heads={n_heads}, n_qubits={n_qubits}, "
            f"n_layers={n_layers}, total_vqc_params={total_vqc}, num_classes={num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]
        Returns:
            [B, num_classes] logits
        """
        head_outputs = [head(x) for head in self.heads]  # K × [B, 2^n_q]
        concat = torch.cat(head_outputs, dim=1)           # [B, K * 2^n_q]
        return self.fusion_net(concat)                    # [B, num_classes]

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, n_heads={self.n_heads}, "
            f"n_qubits={self.n_qubits}, num_classes={self.num_classes}"
        )
