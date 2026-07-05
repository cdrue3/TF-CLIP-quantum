"""
quantum_models/optimisation/quantum_temporal_configurable.py

Configurable Entanglement Temporal VQC — DQAS-inspired Circuit Structure Search.

Same architecture as QuantumTemporalAgg (TQA) but with a configurable
entanglement topology, allowing systematic comparison of circuit structures:

  'full'   : StronglyEntanglingLayers (default — all-to-all + SU(2) rotations)
  'ring'   : Nearest-neighbour CNOT ring: CNOT(i, (i+1) % n_q) per layer
             followed by RY+RZ rotations per qubit
  'linear' : Linear chain CNOT: CNOT(i, i+1) for i in 0..n_q-2
             Most hardware-friendly, lowest depth

This enables a lightweight DQAS-style ablation: find which entanglement
pattern works best for temporal feature aggregation without requiring full
architecture search.

Usage:
    train_qtemporal_ent.py --entanglement full|ring|linear
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumTemporalConfigurable(nn.Module):
    """
    TQA with configurable entanglement topology.

    Args:
        in_features    (int): Feature dimension.
        n_qubits       (int): Qubit count. Default 8.
        n_layers       (int): Number of entanglement layers per frame. Default 2.
        seq_len        (int): T — frames per tracklet. Default 8.
        entanglement   (str): Circuit structure: 'full', 'ring', or 'linear'.
        bypass_quantum (bool): If True, return plain mean-pool.
        device_name    (str): PennyLane device.
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        seq_len: int = 8,
        entanglement: str = "full",
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        assert entanglement in ("full", "ring", "linear"), \
            f"entanglement must be 'full', 'ring', or 'linear', got '{entanglement}'"

        self.in_features    = in_features
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.seq_len        = seq_len
        self.entanglement   = entanglement
        self.n_measurements = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        self.pre_net = nn.Linear(in_features, n_qubits, bias=False)

        if not bypass_quantum:
            n_q = n_qubits
            dev = qml.device(device_name, wires=n_q)

            if entanglement == "full":
                # Standard StronglyEntanglingLayers
                weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_q)
                self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))

                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def _circuit(angles_2d, weights):
                    for t in range(seq_len):
                        qml.AngleEmbedding(angles_2d[t], wires=range(n_q), rotation="Y")
                        qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                    return qml.probs(wires=range(n_q))

            else:
                # Ring or linear: rotation params [n_layers, n_q, 2] (RY + RZ per qubit)
                self.qlayer_weights = nn.Parameter(torch.zeros(n_layers, n_q, 2))

                if entanglement == "ring":
                    @qml.qnode(dev, interface="torch", diff_method="backprop")
                    def _circuit(angles_2d, weights):
                        # angles_2d: [T, B, n_q]; weights: [n_layers, n_q, 2]
                        for t in range(seq_len):
                            qml.AngleEmbedding(angles_2d[t], wires=range(n_q), rotation="Y")
                            for layer in range(n_layers):
                                # Nearest-neighbour ring entanglement
                                for i in range(n_q):
                                    qml.CNOT(wires=[i, (i + 1) % n_q])
                                # Per-qubit rotations
                                for i in range(n_q):
                                    qml.RY(weights[layer, i, 0], wires=i)
                                    qml.RZ(weights[layer, i, 1], wires=i)
                        return qml.probs(wires=range(n_q))

                else:  # linear
                    @qml.qnode(dev, interface="torch", diff_method="backprop")
                    def _circuit(angles_2d, weights):
                        for t in range(seq_len):
                            qml.AngleEmbedding(angles_2d[t], wires=range(n_q), rotation="Y")
                            for layer in range(n_layers):
                                # Linear chain entanglement (most hardware-friendly)
                                for i in range(n_q - 1):
                                    qml.CNOT(wires=[i, i + 1])
                                for i in range(n_q):
                                    qml.RY(weights[layer, i, 0], wires=i)
                                    qml.RZ(weights[layer, i, 1], wires=i)
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
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, entanglement={self.entanglement}, "
            f"bypass={self.bypass_quantum}"
        )
