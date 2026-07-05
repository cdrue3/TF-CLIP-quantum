"""
quantum_models/angle/quantum_temporal_reupload.py

Light Re-uploading Temporal Quantum Aggregation.

Fixes the intractable QClassifierReupload (96 blocks, ~57s/batch) by using
n_reupload=4 re-upload blocks with INDEPENDENT VQC weights per block.

Unlike TQA (which shares one weight tensor across all T uploads), each
re-upload block k has its own StronglyEntanglingLayers weight tensor. This
is the survey's data re-uploading classifier applied to temporal features:
more quantum expressivity without the barren plateau risk of deeper circuits.

Architecture:
    x [B, T, D]
    → pre_net: Linear(D → n_qubits, bias=False)  [shared compression]
    → sigmoid(·) * π → angles [B, T, n_q]
    → VQC circuit with n_reupload independent weight blocks:
          for t in range(T):
              AngleEmbedding(angles[t])
              StronglyEntanglingLayers(weights[t % n_reupload])  ← independent per block
      → probs [B, 2^n_q]
    → upscale: Linear(2^n_q → D)
    → output = mean_pool(x) + delta

Total circuit params: n_reupload × n_layers × n_qubits × 3
  = 4 × 2 × 8 × 3 = 192  (vs 48 for standard TQA)

bypass_quantum=True: returns x.mean(1).
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumTemporalReupload(nn.Module):
    """
    Light re-uploading TQA with independent VQC weights per block.

    Args:
        in_features  (int): Feature dimension. Default 768.
        n_qubits     (int): Qubit count. Default 8.
        n_layers     (int): SEL depth per block. Default 2.
        seq_len      (int): T — frames per tracklet. Default 8.
        n_reupload   (int): Number of independent VQC weight blocks. Default 4.
        bypass_quantum (bool): If True, return plain mean-pool.
        device_name  (str): PennyLane device.
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        seq_len: int = 8,
        n_reupload: int = 4,
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features    = in_features
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.seq_len        = seq_len
        self.n_reupload     = n_reupload
        self.n_measurements = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        self.pre_net = nn.Linear(in_features, n_qubits, bias=False)

        if not bypass_quantum:
            n_q = n_qubits
            dev = qml.device(device_name, wires=n_q)
            weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_q)

            # n_reupload independent weight tensors stacked as a single Parameter
            # Shape: [n_reupload, n_layers, n_q, 3]
            self.qlayer_weights = nn.Parameter(
                torch.zeros(n_reupload, *weight_shape)
            )

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _circuit(angles_2d, weights_3d):
                # angles_2d: [T, B, n_q]
                # weights_3d: [n_reupload, n_layers, n_q, 3]
                for t in range(seq_len):
                    block_idx = t % n_reupload
                    qml.AngleEmbedding(angles_2d[t], wires=range(n_q), rotation="Y")
                    qml.StronglyEntanglingLayers(weights_3d[block_idx], wires=range(n_q))
                return qml.probs(wires=range(n_q))

            self.circuit = _circuit

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

        angles = torch.sigmoid(self.pre_net(x.float().reshape(B * T, D))) * math.pi
        angles = angles.reshape(B, T, self.n_qubits)  # [B, T, n_q]

        angles_f  = angles.permute(1, 0, 2).float()   # [T, B, n_q]
        weights_f = self.qlayer_weights.float()        # [n_reupload, n_layers, n_q, 3]

        q_out = self.circuit(angles_f, weights_f).float()  # [B, 2^n_q]
        delta = self.upscale(q_out)

        return (mean_feat.float() + delta).to(dtype=input_dtype)

    def extra_repr(self) -> str:
        total_q_params = self.n_reupload * self.n_layers * self.n_qubits * 3
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, seq_len={self.seq_len}, "
            f"n_reupload={self.n_reupload}, vqc_params={total_q_params}, "
            f"bypass={self.bypass_quantum}"
        )
