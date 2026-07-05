"""
quantum_models/angle/quantum_temporal_dense.py

Dense Angle Encoding Temporal Quantum Aggregation.

Identical to QuantumTemporalAgg (TQA) but uses dense angle encoding:
  - pre_net: Linear(in_features → 2*n_qubits)   [double output]
  - Circuit: AngleEmbedding(angles[:n_q], rotation='Y')   [RY per qubit]
             AngleEmbedding(angles[n_q:], rotation='Z')   [RZ per qubit]

This encodes 2 features per qubit instead of 1, doubling the information
density without changing circuit depth or qubit count. Addresses the
768→8 compression bottleneck documented in the survey.

Total parameters: same VQC weights (n_layers, n_q, 3), but pre_net is
Linear(D, 2*n_q) instead of Linear(D, n_q) — 2x more classical pre-net params.

bypass_quantum=True: returns x.mean(1) directly (classical mean-pool ablation).
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumTemporalDense(nn.Module):
    """
    Dense-Angle Temporal Quantum Aggregation: [B, T, in_features] → [B, in_features].

    Data re-uploading over T frames with dense angle encoding (RY + RZ per qubit).
    Each qubit receives two features: one via RY rotation, one via RZ rotation.
    pre_net maps D → 2*n_qubits; first half → Y rotations, second half → Z rotations.

    Skip connection: output = mean_pool(x) + upscale(VQC(pre_net(x))).
    At init, upscale is near-zero → output ≈ mean_pool(x).

    Args:
        in_features    (int): Feature dimension (e.g. 768 for ViT-B-16).
        n_qubits       (int): Qubit count. Default 8 → 16 features encoded (2 per qubit).
        n_layers       (int): StronglyEntanglingLayers depth. Default 2.
        seq_len        (int): T — frames per tracklet. Default 8.
        bypass_quantum (bool): If True, return plain mean-pool.
        device_name    (str): PennyLane device. Default 'default.qubit' (CPU sim).
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        seq_len: int = 8,
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features    = in_features
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.seq_len        = seq_len
        self.n_measurements = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        # Dense pre-net: D → 2*n_qubits  (first n_q → RY angles, second n_q → RZ angles)
        self.pre_net = nn.Linear(in_features, 2 * n_qubits, bias=False)

        if not bypass_quantum:
            n_q = n_qubits
            dev = qml.device(device_name, wires=n_q)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _circuit(angles_2d, weights):
                # angles_2d: [T, B, 2*n_q]
                # weights:   [n_layers, n_q, 3]
                for t in range(seq_len):
                    # Dense encoding: split angles into Y and Z halves
                    qml.AngleEmbedding(angles_2d[t, :, :n_q], wires=range(n_q), rotation="Y")
                    qml.AngleEmbedding(angles_2d[t, :, n_q:], wires=range(n_q), rotation="Z")
                    qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                return qml.probs(wires=range(n_q))

            self.circuit = _circuit
            weight_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=n_layers, n_wires=n_q
            )
            self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))

        self.upscale = nn.Linear(self.n_measurements, in_features, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.01)
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, in_features]
        Returns:
            [B, in_features]
        """
        mean_feat = x.mean(1)

        if self.bypass_quantum:
            return mean_feat

        input_dtype = x.dtype
        B, T, D = x.shape

        # pre_net → [B, T, 2*n_q]; scale to (0, π)
        angles = torch.sigmoid(self.pre_net(x.float().reshape(B * T, D))) * math.pi
        angles = angles.reshape(B, T, 2 * self.n_qubits)   # [B, T, 2*n_q]

        angles_f  = angles.permute(1, 0, 2).float()  # [T, B, 2*n_q]
        weights_f = self.qlayer_weights.float()

        q_out = self.circuit(angles_f, weights_f).float()  # [B, 2^n_q]
        delta = self.upscale(q_out)

        return (mean_feat.float() + delta).to(dtype=input_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, seq_len={self.seq_len}, "
            f"encoding=dense_angle (RY+RZ, 2 features/qubit), bypass={self.bypass_quantum}"
        )
