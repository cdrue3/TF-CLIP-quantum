"""
quantum_models/angle/quantum_temporal_deep.py

Deep VQC Temporal Quantum Aggregation (n_layers=4).

Identical to QuantumTemporalAgg (TQA) but with default n_layers=4 and
tighter near-identity initialization (std=0.005 instead of 0.01).

The increased circuit depth (48→96 trainable VQC params) raises barren
plateau risk. Mitigated by:
  - tighter init (std=0.005): starts even closer to identity circuit
  - gradient norm clipping applied at training-script level (--gradient_clip)

Use train_qtemporal_deep.py which monitors gradient norms and clips them.
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumTemporalDeep(nn.Module):
    """
    Deep TQA: same as QuantumTemporalAgg but default n_layers=4.

    Args:
        in_features    (int): Feature dimension.
        n_qubits       (int): Qubit count. Default 8.
        n_layers       (int): VQC depth. Default 4 (double of standard TQA).
        seq_len        (int): T — frames per tracklet. Default 8.
        bypass_quantum (bool): If True, return plain mean-pool.
        device_name    (str): PennyLane device.
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 8,
        n_layers: int = 4,
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

        self.pre_net = nn.Linear(in_features, n_qubits, bias=False)

        if not bypass_quantum:
            n_q = n_qubits
            dev = qml.device(device_name, wires=n_q)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _circuit(angles_2d, weights):
                # angles_2d: [T, B, n_q]; weights: [n_layers, n_q, 3]
                for t in range(seq_len):
                    qml.AngleEmbedding(angles_2d[t], wires=range(n_q), rotation="Y")
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
            # Tighter than standard TQA (0.01) to reduce barren plateau risk at depth 4
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.005)
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_feat = x.mean(1)
        if self.bypass_quantum:
            return mean_feat

        input_dtype  = x.dtype
        input_device = x.device
        B, T, D = x.shape

        angles = torch.sigmoid(self.pre_net(x.float().reshape(B * T, D))) * math.pi
        angles = angles.reshape(B, T, self.n_qubits)

        angles_cpu  = angles.permute(1, 0, 2).float()
        weights_cpu = self.qlayer_weights.float()

        q_out = self.circuit(angles_cpu, weights_cpu).float()
        delta = self.upscale(q_out)

        return (mean_feat.float() + delta).to(input_dtype)

    def extra_repr(self) -> str:
        vqc_params = self.n_layers * self.n_qubits * 3
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, seq_len={self.seq_len}, "
            f"vqc_params={vqc_params}, init_std=0.005, bypass={self.bypass_quantum}"
        )
