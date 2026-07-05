"""
quantum_models/feature_extraction/quantum_autoencoder.py

Quantum Autoencoder Feature Extractor (QPCA-inspired).

Applied to pooled ViT features [B, 768] after temporal mean-pooling.
A VQC acts as the bottleneck of an autoencoder:

Architecture:
    x [B, 768]
    → encoder_pre: Linear(768 → n_qubits, bias=False)
    → sigmoid(·) * π → VQC → probs [B, 2^n_qubits]    ← quantum bottleneck
    → decoder: Linear(2^n_qubits → 768, bias=False)    → x_recon [B, 768]

The quantum bottleneck IS the compressed representation. Training adds a
reconstruction loss: MSE(x_recon, x.detach()) * recon_weight, forcing the
VQC to preserve information about the input features.

At inference, the module returns x_recon as the processed feature (the
VQC-compressed-then-decoded representation). The reconstruction loss is only
computed during training.

Skip connection: output = x + x_recon  (residual — same pattern as other VQCs)
At init: decoder.weight ≈ 0 → x_recon ≈ 0 → output ≈ x (classical baseline).

bypass_quantum=True: returns x unchanged, no reconstruction loss.
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumAutoEncoder(nn.Module):
    """
    VQC-bottleneck autoencoder for feature compression/enhancement.

    Returns both the reconstructed features (for the main pipeline) and
    optionally the reconstruction loss (for training-time regularization).

    Args:
        in_features   (int): Input feature dimension. Default 768.
        n_qubits      (int): Qubit count (bottleneck size = 2^n_qubits). Default 6.
        n_layers      (int): VQC depth. Default 2.
        bypass_quantum(bool): If True, return x unchanged.
        device_name   (str): PennyLane device.
    """

    def __init__(
        self,
        in_features: int = 768,
        n_qubits: int = 6,
        n_layers: int = 2,
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features    = in_features
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.n_measurements = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        # Encoder: projects features to angle space
        self.encoder_pre = nn.Linear(in_features, n_qubits, bias=False)

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

        # Decoder: reconstructs features from quantum bottleneck representation
        self.decoder = nn.Linear(self.n_measurements, in_features, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.encoder_pre.weight, a=0, mode="fan_in")
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.01)
        # Near-zero decoder: output ≈ x at init (residual identity)
        nn.init.normal_(self.decoder.weight, mean=0, std=0.001)

    def _apply(self, fn):
        super()._apply(fn)
        if not self.bypass_quantum:
            self.qlayer_weights.data = self.qlayer_weights.data.cpu().float()
        return self

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [B, in_features]
        Returns:
            (output [B, in_features], x_recon [B, in_features] or None)
            x_recon is used to compute reconstruction loss during training.
            output = x + x_recon  (residual)
        """
        if self.bypass_quantum:
            return x, None

        input_dtype  = x.dtype
        input_device = x.device

        angles = torch.sigmoid(self.encoder_pre(x.float())) * math.pi  # [B, n_q]
        angles_cpu  = angles.cpu().float()
        weights_cpu = self.qlayer_weights.cpu().float()

        probs = self.circuit(angles_cpu, weights_cpu).float().to(input_device)  # [B, 2^n_q]
        x_recon = self.decoder(probs)  # [B, in_features]

        output = (x.float() + x_recon).to(input_dtype)
        return output, x_recon.to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"bottleneck_dim={self.n_measurements}, bypass={self.bypass_quantum}"
        )
