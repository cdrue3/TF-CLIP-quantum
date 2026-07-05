"""
quantum_models/quantum_temporal_diff.py

Quantum Temporal Difference (QTD).

Encodes *motion* rather than absolute frame content.
Feed frame differences Δ_t = frame_{t+1} - frame_t (T-1 difference vectors)
through a shared VQC, then apply a residual on mean_pool.

Architecture:
    Input x: [B, T, in_features]
    1. diffs = x[:, 1:] - x[:, :-1]          → [B, T-1, in_features]
    2. pre_net: Linear(in_features → n_qubits) [per difference via reshape]
    3. sigmoid(·) * π                          → angles ∈ (0, π)
    4. VQC circuit (T-1 baked in as closure, shared weights [n_layers, n_qubits, 3]):
           for t in range(T-1):
               AngleEmbedding(angles[t], wires=..., rotation='Y')
               StronglyEntanglingLayers(weights, wires=...)
       qml.probs(wires=...) → [2^n_qubits]
    5. upscale: Linear(2^n_qubits → in_features, bias=False)  — init N(0, 0.001)
    6. Skip: output = mean_pool(x) + delta    (starts as plain mean-pool at init)
    Output: [B, in_features]

Motivation: mean-pool and TQA both encode absolute frame content; QTD exclusively
encodes *change* between frames, providing complementary motion information.

bypass_quantum=True: returns x.mean(1) directly (classical mean-pool ablation).
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumTemporalDiff(nn.Module):
    """
    Quantum Temporal Difference: [B, T, in_features] → [B, in_features].

    Computes T-1 frame differences, uploads them sequentially into a shared VQC
    (data re-uploading over motion), then applies a residual on mean_pool.

    Skip connection: output = mean_pool(x) + upscale(VQC(pre_net(diffs))).
    At init, upscale is near-zero → output ≈ mean_pool(x).

    bypass_quantum=True: output = x.mean(1) — exact classical mean-pool (for ablation).

    Args:
        in_features    (int): Feature dimension (e.g. 768 for ViT-B-16).
        n_qubits       (int): Qubit count. Default 8 → 256 probability outcomes.
        n_layers       (int): StronglyEntanglingLayers depth. Default 2.
        seq_len        (int): T — frames per tracklet (baked into circuit). Default 4.
                              VQC processes T-1 differences.
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
        self.n_diffs        = seq_len - 1          # T-1 = 3 for seq_len=4
        self.n_measurements = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        # Pre-net: compress each difference vector from in_features to n_qubits angles.
        self.pre_net = nn.Linear(in_features, n_qubits, bias=False)

        if not bypass_quantum:
            n_q    = n_qubits
            n_diff = self.n_diffs
            dev    = qml.device(device_name, wires=n_q)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _circuit(angles_2d, weights):
                # angles_2d: [T-1, B, n_q] — row t = difference t's angles for all B samples
                # weights:   [n_layers, n_q, 3] — shared across all T-1 differences
                # PennyLane broadcasts over B via parameter broadcasting.
                for t in range(n_diff):
                    qml.AngleEmbedding(angles_2d[t], wires=range(n_q), rotation="Y")
                    qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                return qml.probs(wires=range(n_q))

            self.circuit = _circuit
            weight_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=n_layers, n_wires=n_q
            )
            self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))

        # Upscale: 2^n_qubits → in_features.
        # Near-zero init so skip starts as plain mean-pool.
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
            x: [B, T, in_features]  (float16 or float32, any device)
        Returns:
            [B, in_features]  (same dtype/device as input)
        """
        mean_feat = x.mean(1)   # [B, in_features] — always computed for skip

        if self.bypass_quantum:
            return mean_feat

        input_dtype = x.dtype
        B, T, D = x.shape

        # Frame differences: [B, T-1, in_features]
        diffs = x[:, 1:, :] - x[:, :-1, :]   # [B, T-1, D]

        # Pre-net per difference → [B, T-1, n_qubits] angles
        angles = torch.sigmoid(self.pre_net(diffs.float().reshape(B * self.n_diffs, D))) * math.pi
        angles = angles.reshape(B, self.n_diffs, self.n_qubits)   # [B, T-1, n_q]

        # Transpose to [T-1, B, n_q] so circuit[t] = [B, n_q] → PennyLane broadcasts over B.
        angles_f  = angles.permute(1, 0, 2).float()  # [T-1, B, n_q]
        weights_f = self.qlayer_weights.float()

        q_out = self.circuit(angles_f, weights_f).float()  # [B, 2^n_q]

        delta = self.upscale(q_out)   # [B, in_features]

        return (mean_feat.float() + delta).to(dtype=input_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, seq_len={self.seq_len}, "
            f"n_diffs={self.n_diffs}, n_measurements={self.n_measurements}, "
            f"bypass={self.bypass_quantum}"
        )
