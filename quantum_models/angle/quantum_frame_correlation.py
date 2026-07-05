"""
quantum_models/quantum_frame_correlation.py

Quantum Frame Correlation (QFC).

Processes all T(T-1)/2 pairs of frames jointly — the VQC sees concatenated
[frame_i || frame_j] inputs, encoding cross-frame quantum correlations.

Architecture:
    Input x: [B, T, in_features]
    1. Build all pairs: T(T-1)/2 = 6 pairs for T=4
       pair_feats: [B, n_pairs, 2*in_features]
    2. pre_net: Linear(2*in_features → n_qubits, bias=False)  [per pair]
    3. sigmoid(·) * π                                          → angles ∈ (0, π)
    4. VQC: AngleEmbedding + StronglyEntanglingLayers
       Batched: run all B*n_pairs through circuit at once (reshape → [n_pairs*B, n_q])
       → probs [B*n_pairs, 2^n_qubits]
    5. Reshape → [B, n_pairs, 2^n_qubits], mean over pairs → [B, 2^n_qubits]
    6. upscale: Linear(2^n_qubits → in_features, bias=False)   → delta [B, in_features]
    7. output = mean_pool(x) + delta

Motivation: genuinely captures cross-frame quantum interference — the joint encoding
of two frames into the same circuit allows interference patterns impossible with
per-frame processing (TQA) or frame differences (QTD).

bypass_quantum=True: returns x.mean(1) directly (classical mean-pool ablation).
"""

import math
from itertools import combinations

import torch
import torch.nn as nn
import pennylane as qml


class QuantumFrameCorrelation(nn.Module):
    """
    Quantum Frame Correlation: [B, T, in_features] → [B, in_features].

    Processes T(T-1)/2 frame pairs jointly through a single VQC.
    Averages pair outputs, then applies residual on mean_pool.

    output = mean_pool(x) + upscale(mean_over_pairs(VQC(pre_net([frame_i || frame_j]))))

    bypass_quantum=True: output = x.mean(1) — exact classical mean-pool.

    Args:
        in_features    (int): Feature dimension (e.g. 768 for ViT-B-16).
        n_qubits       (int): Qubit count. Default 8 → 256 probability outcomes.
        n_layers       (int): StronglyEntanglingLayers depth. Default 2.
        seq_len        (int): T — frames per tracklet. Default 4 → 6 pairs.
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
        self.n_pairs        = seq_len * (seq_len - 1) // 2   # T(T-1)/2 = 6 for T=4
        self.n_measurements = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        # Precompute pair indices (fixed — depends only on seq_len).
        pair_idx = list(combinations(range(seq_len), 2))
        self.register_buffer(
            "pair_i", torch.tensor([p[0] for p in pair_idx], dtype=torch.long)
        )
        self.register_buffer(
            "pair_j", torch.tensor([p[1] for p in pair_idx], dtype=torch.long)
        )

        # Pre-net: joint pair encoding 2*in_features → n_qubits angles.
        self.pre_net = nn.Linear(2 * in_features, n_qubits, bias=False)

        if not bypass_quantum:
            n_q = n_qubits
            dev = qml.device(device_name, wires=n_q)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _circuit(inputs, weights):
                # inputs:  [B*n_pairs, n_q] — PennyLane broadcasts over batch dim
                # weights: [n_layers, n_q, 3]
                qml.AngleEmbedding(inputs, wires=range(n_q), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                return qml.probs(wires=range(n_q))

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            self.qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

        # Upscale: 2^n_qubits → in_features. Near-zero init → delta ≈ 0 at init.
        self.upscale = nn.Linear(self.n_measurements, in_features, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer.weights, mean=0, std=0.01)
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)

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

        input_dtype  = x.dtype
        B, T, D = x.shape
        x_f = x.float()

        # Build pair features: [B, n_pairs, 2*in_features]
        frames_i = x_f[:, self.pair_i, :]   # [B, n_pairs, D]
        frames_j = x_f[:, self.pair_j, :]   # [B, n_pairs, D]
        pair_feats = torch.cat([frames_i, frames_j], dim=-1)   # [B, n_pairs, 2D]

        # Pre-net: [B, n_pairs, 2D] → [B, n_pairs, n_q] angles
        angles = torch.sigmoid(
            self.pre_net(pair_feats.reshape(B * self.n_pairs, 2 * D))
        ) * math.pi
        angles = angles.reshape(B * self.n_pairs, self.n_qubits)   # [B*n_pairs, n_q]

        # VQC: single batched call over all B*n_pairs.
        q_out = self.qlayer(angles.float())                        # [B*n_pairs, 2^n_q]

        # Average over pairs: [B, n_pairs, 2^n_q] → [B, 2^n_q]
        q_out = q_out.reshape(B, self.n_pairs, self.n_measurements).mean(1)

        delta = self.upscale(q_out)   # [B, in_features]

        return (mean_feat.float() + delta).to(dtype=input_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, seq_len={self.seq_len}, "
            f"n_pairs={self.n_pairs}, n_measurements={self.n_measurements}, "
            f"bypass={self.bypass_quantum}"
        )
