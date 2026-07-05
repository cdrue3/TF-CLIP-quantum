"""
quantum_models/angle/quantum_pca_preprocess.py

Quantum Channel Preprocessing (QPCA-inspired).

Applied to raw image tensors [B*T, 3, H, W] BEFORE the ViT backbone — the
"pre-processing" stage in the survey's pipeline taxonomy.

Inspired by QPCA's ability to learn compact quantum-enhanced representations
of input data, this module learns channel-wise attention weights via a small
VQC. The 3 RGB channels are summarised via global average pooling, compressed
into n_qubits (default 4) quantum angles, processed by a VQC, and the output
produces per-channel attention scalars that multiplicatively rescale the image.

Architecture:
    x [B*T, 3, H, W]
    → global_avg_pool(x) → [B*T, 3]
    → pre_net: Linear(3 → n_qubits=4, bias=True)
    → sigmoid(·) * π → VQC → probs [B*T, 2^4=16]
    → channel_net: Linear(16 → 3) + sigmoid → channel_weights [B*T, 3] ∈ (0,1)
    → output = x * (1 + channel_weights[:, :, None, None])

Residual design: channel_weights ≈ 0 at init → output ≈ x (identity).
n_qubits=4 keeps simulation fast (2^4=16 states, O(16) per forward).

bypass_quantum=True: returns x unchanged.
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumChannelPreprocess(nn.Module):
    """
    Quantum channel-wise preprocessing: learns RGB attention via a small VQC.

    Runs at both train AND eval time (unlike adapter-style modules).
    Applied to raw image pixels before ViT patch embedding.

    Args:
        n_channels    (int): Input channels. Default 3 (RGB).
        n_qubits      (int): Qubit count. Default 4 → 16 probs, fast simulation.
        n_layers      (int): VQC depth. Default 1.
        bypass_quantum(bool): If True, return input unchanged.
        device_name   (str): PennyLane device.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_qubits: int = 4,
        n_layers: int = 1,
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.n_channels     = n_channels
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
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

        # channel_net: maps VQC probs → channel attention scalars
        self.channel_net = nn.Linear(self.n_measurements, n_channels)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        nn.init.zeros_(self.pre_net.bias)
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.01)
        # channel_net init: output near 0 → attention near 0 → output ≈ input (identity)
        nn.init.normal_(self.channel_net.weight, mean=0, std=0.001)
        nn.init.zeros_(self.channel_net.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*T, 3, H, W]
        Returns:
            [B*T, 3, H, W]  (channel-rescaled, same shape)
        """
        if self.bypass_quantum:
            return x

        input_dtype  = x.dtype
        input_device = x.device

        # Global average pool → [B*T, 3] channel descriptor
        channel_desc = x.float().mean(dim=[2, 3])  # [B*T, 3]

        angles = torch.sigmoid(self.pre_net(channel_desc)) * math.pi  # [B*T, n_q]
        angles_cpu  = angles.cpu().float()
        weights_cpu = self.qlayer_weights.float()

        probs = self.circuit(angles_cpu, weights_cpu).float()  # [B*T, 2^n_q]

        # channel_weights: [B*T, 3] ∈ ≈0 at init → identity residual
        channel_weights = self.channel_net(probs)  # [B*T, 3]
        channel_weights = channel_weights[:, :, None, None]  # [B*T, 3, 1, 1] for broadcast

        return (x.float() * (1.0 + channel_weights)).to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"n_channels={self.n_channels}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, bypass={self.bypass_quantum}"
        )
