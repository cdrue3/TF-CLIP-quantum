"""
quantum_models/quantum_temporal_gated.py

Quantum-Gated Temporal (QGT).

Builds on TQA: a learned scalar gate controls how much of the VQC correction to apply,
conditioned on the mean-pooled tracklet.

Architecture:
    Input x: [B, T, in_features]
    1. mean_feat = x.mean(1)                                    [B, in_features]
    2. pre_net: Linear(in_features → n_qubits, bias=False)      per frame
    3. sigmoid(·) * π                                           angles ∈ (0, π)
    4. VQC (same as TQA — T frames, shared weights):
           for t in range(T):
               AngleEmbedding + StronglyEntanglingLayers
       → probs [B, 2^n_qubits]
    5. upscale: Linear(2^n_qubits → in_features, bias=False)    → delta [B, in_features]
    6. gate_net: Linear(in_features → 1, bias=True) + sigmoid   → g ∈ (0,1) scalar per sample
    7. output = mean_feat + g * delta

Gate interpretation: g≈0 → trust mean-pool, g≈1 → apply full VQC correction.
Samples with complex temporal dynamics should learn high g; static tracklets → low g.

bypass_quantum=True: returns x.mean(1) directly (classical mean-pool ablation).
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumTemporalGated(nn.Module):
    """
    Quantum-Gated Temporal Aggregation: [B, T, in_features] → [B, in_features].

    output = mean_pool(x) + gate(mean_pool(x)) * upscale(VQC(pre_net(x)))

    The gate is a learned scalar per sample — interpretable: log g to see which
    tracklets benefit from quantum temporal correction.

    bypass_quantum=True: output = x.mean(1) — exact classical mean-pool.

    Args:
        in_features    (int): Feature dimension (e.g. 768 for ViT-B-16).
        n_qubits       (int): Qubit count. Default 8 → 256 probability outcomes.
        n_layers       (int): StronglyEntanglingLayers depth. Default 2.
        seq_len        (int): T — frames per tracklet (baked into circuit). Default 4.
        bypass_quantum (bool): If True, skip VQC and return plain mean-pool.
        device_name    (str): PennyLane device. Default 'default.qubit' (CPU sim).
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        seq_len: int = 4,
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

        # Pre-net: compress each frame from in_features to n_qubits angles.
        self.pre_net = nn.Linear(in_features, n_qubits, bias=False)

        if not bypass_quantum:
            n_q  = n_qubits
            dev  = qml.device(device_name, wires=n_q)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _circuit(angles_2d, weights):
                # angles_2d: [T, B, n_q] — row t = frame t's angles for all B samples
                # weights:   [n_layers, n_q, 3] — shared across all T frames
                for t in range(seq_len):
                    qml.AngleEmbedding(angles_2d[t], wires=range(n_q), rotation="Y")
                    qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                return qml.probs(wires=range(n_q))

            self.circuit = _circuit
            weight_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=n_layers, n_wires=n_q
            )
            self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))

        # Upscale: 2^n_qubits → in_features. Near-zero init → delta ≈ 0 at init.
        self.upscale = nn.Linear(self.n_measurements, in_features, bias=False)

        # Gate: mean_pool → scalar ∈ (0,1). Init bias=-2 → sigmoid(-2)≈0.12 (small gate at init).
        self.gate_net = nn.Linear(in_features, 1, bias=True)

        # Stores gate values from last forward pass — accessible for analysis.
        self.last_gates = None

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.01)
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)
        nn.init.normal_(self.gate_net.weight, mean=0, std=0.01)
        nn.init.constant_(self.gate_net.bias, -2.0)   # gate starts near 0.12 (nearly closed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, in_features]  (float16 or float32, any device)
        Returns:
            [B, in_features]  (same dtype/device as input)
        """
        mean_feat = x.mean(1)   # [B, in_features]

        if self.bypass_quantum:
            return mean_feat

        input_dtype = x.dtype
        B, T, D = x.shape

        # Gate: scalar per sample conditioned on mean-pooled tracklet.
        g = torch.sigmoid(self.gate_net(mean_feat.float()))   # [B, 1]
        self.last_gates = g.squeeze(1).detach().cpu()         # [B] — stored for analysis

        # Pre-net per frame → [B, T, n_qubits] angles
        angles = torch.sigmoid(self.pre_net(x.float().reshape(B * T, D))) * math.pi
        angles = angles.reshape(B, T, self.n_qubits)          # [B, T, n_q]

        # Batched VQC: [T, B, n_q]
        angles_f  = angles.permute(1, 0, 2).float()
        weights_f = self.qlayer_weights.float()
        q_out = self.circuit(angles_f, weights_f).float()   # [B, 2^n_q]

        delta = self.upscale(q_out)   # [B, in_features]

        return (mean_feat.float() + g * delta).to(dtype=input_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, seq_len={self.seq_len}, "
            f"n_measurements={self.n_measurements}, bypass={self.bypass_quantum}"
        )
