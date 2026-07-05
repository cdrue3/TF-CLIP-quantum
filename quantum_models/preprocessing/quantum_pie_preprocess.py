"""
quantum_models/angle/quantum_pie_preprocess.py

Quantum Probability Image Encoding (QPIE-inspired) Spatial Filter.

Applied to raw image tensors [B*T, 3, H, W] BEFORE the ViT backbone.

QPIE encodes pixel intensities as probability amplitudes. Here we implement
a spatial quantum filter: the image is divided into non-overlapping spatial
grid cells, and a shared 2-qubit VQC processes each cell's mean RGB values,
producing a learned spatial attention map that gates the image.

Architecture:
    x [B*T, 3, H, W]
    → divide into G×G spatial grid cells → [B*T, G*G, 3]
    → pre_net: Linear(3 → n_qubits=2, bias=True) per cell
    → sigmoid(·) * π → VQC → probs [B*T*G*G, 2^2=4]
    → spatial_net: Linear(4 → 1) + sigmoid → gate [B*T, G*G] ∈ (0,1)
    → upsample gate to [B*T, 1, H, W]
    → output = x * (1 + gate_map)

n_qubits=2: minimal circuit, very fast (2^2=4 states). Shared across all cells.
Default grid_size=4: 4×4=16 spatial cells, tractable even on large images.

bypass_quantum=True: returns x unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuantumSpatialFilter(nn.Module):
    """
    QPIE-inspired quantum spatial filter on raw images.

    Learns a spatial attention map via a 2-qubit VQC applied per grid cell.

    Args:
        n_channels    (int): Input channels. Default 3 (RGB).
        n_qubits      (int): Qubit count per cell VQC. Default 2 (fast).
        n_layers      (int): VQC depth. Default 1.
        grid_size     (int): Spatial grid divisions per dimension. Default 4 (→ 4×4=16 cells).
        bypass_quantum(bool): If True, return input unchanged.
        device_name   (str): PennyLane device.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_qubits: int = 2,
        n_layers: int = 1,
        grid_size: int = 4,
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.n_channels     = n_channels
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.grid_size      = grid_size
        self.n_cells        = grid_size * grid_size
        self.n_measurements = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        self.pre_net = nn.Linear(n_channels, n_qubits, bias=True)

        if not bypass_quantum:
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

        # Maps VQC probs → scalar gate per cell
        self.spatial_net = nn.Linear(self.n_measurements, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        nn.init.zeros_(self.pre_net.bias)
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.01)
        # Near-zero init → gate ≈ 0 → output ≈ input
        nn.init.normal_(self.spatial_net.weight, mean=0, std=0.001)
        nn.init.zeros_(self.spatial_net.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*T, 3, H, W]
        Returns:
            [B*T, 3, H, W]
        """
        if self.bypass_quantum:
            return x

        input_dtype  = x.dtype
        input_device = x.device
        BT, C, H, W  = x.shape
        G = self.grid_size

        # Divide image into G×G cells via adaptive average pool → [BT, 3, G, G]
        cell_means = F.adaptive_avg_pool2d(x.float(), (G, G))  # [BT, 3, G, G]
        # Reshape to [BT*G*G, 3] for batch VQC processing
        cell_means = cell_means.permute(0, 2, 3, 1).reshape(BT * G * G, C)  # [BT*G^2, 3]

        angles = torch.sigmoid(self.pre_net(cell_means)) * math.pi  # [BT*G^2, n_q]
        angles_cpu  = angles.cpu().float()
        weights_cpu = self.qlayer_weights.float()

        probs = self.circuit(angles_cpu, weights_cpu).float()  # [BT*G^2, 2^n_q]

        # Spatial gate: [BT*G^2, 1] → [BT, 1, G, G]
        gate = self.spatial_net(probs)  # [BT*G^2, 1]
        gate = gate.reshape(BT, 1, G, G)  # [BT, 1, G, G]

        # Upsample gate to original spatial resolution
        gate_map = F.interpolate(gate, size=(H, W), mode='bilinear', align_corners=False)  # [BT,1,H,W]

        return (x.float() * (1.0 + gate_map)).to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"n_channels={self.n_channels}, n_qubits={self.n_qubits}, "
            f"grid_size={self.grid_size}, n_cells={self.n_cells}, bypass={self.bypass_quantum}"
        )
