"""
quantum_models/quantum_temporal.py

Temporal Quantum Aggregation (TQA).

Replaces the mean-pool temporal aggregation [B, T, 768] → [B, 768] with a VQC
that uploads T frames sequentially (data re-uploading over time).

Architecture:
    Input x: [B, T, in_features]
    1. pre_net: Linear(in_features → n_qubits, bias=False)  [applied per frame via reshape]
    2. sigmoid(·) * π — scale to angle range (0, π)
    3. Flatten: [B, T, n_qubits] → [B, T * n_qubits]  for TorchLayer batching
    4. VQC circuit (T baked in as closure, shared weights [n_layers, n_qubits, 3]):
           for t in range(T):
               AngleEmbedding(inputs[t*n_q:(t+1)*n_q], wires=..., rotation='Y')
               StronglyEntanglingLayers(weights, wires=...)   # shared across frames
       qml.probs(wires=...) → [2^n_qubits]
    5. upscale: Linear(2^n_qubits → in_features, bias=False)  — init N(0, 0.001)
    6. Skip: output = mean_pool(x) + delta  (starts as plain mean-pool at init)
    Output: [B, in_features]

bypass_quantum=True: returns x.mean(1) directly (classical mean-pool ablation).
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumTemporalAgg(nn.Module):
    """
    Temporal Quantum Aggregation: [B, T, in_features] → [B, in_features].

    Data re-uploading over time: T frames are uploaded sequentially into a shared
    VQC, each frame's angles followed by an entangling block. This allows temporal
    interference between frames, unlike mean-pooling which treats frames independently.

    Skip connection: output = mean_pool(x) + upscale(VQC(pre_net(x))).
    At init, upscale is near-zero → output ≈ mean_pool(x). VQC learns to add corrections.

    bypass_quantum=True: output = x.mean(1) — exact classical mean-pool (for ablation).

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
        self.in_features   = in_features
        self.n_qubits      = n_qubits
        self.n_layers      = n_layers
        self.seq_len       = seq_len
        self.n_measurements = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        # Pre-net: compress each frame from in_features to n_qubits angles.
        self.pre_net = nn.Linear(in_features, n_qubits, bias=False)

        if not bypass_quantum:
            # VQC: T frames uploaded sequentially, shared entangling weights.
            # Circuit takes [T, n_q] angles (one row per frame) — avoids slice
            # indexing issues inside qnode. Batching handled manually in forward().
            n_q = n_qubits
            dev = qml.device(device_name, wires=n_q)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _circuit(angles_2d, weights):
                # angles_2d: [T, B, n_q] — row t = frame t's angles for all B samples
                # weights:   [n_layers, n_q, 3] — shared across all T frames
                # PennyLane broadcasts over B via parameter broadcasting.
                for t in range(seq_len):
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

        # Pre-net per frame → [B, T, n_qubits] angles
        angles = torch.sigmoid(self.pre_net(x.float().reshape(B * T, D))) * math.pi
        angles = angles.reshape(B, T, self.n_qubits)   # [B, T, n_q]

        # Transpose to [T, B, n_q] so circuit[t] = [B, n_q] → PennyLane broadcasts over B.
        angles_f  = angles.permute(1, 0, 2).float()  # [T, B, n_q]
        weights_f = self.qlayer_weights.float()

        q_out = self.circuit(angles_f, weights_f).float()  # [B, 2^n_q]

        delta = self.upscale(q_out)   # [B, in_features]

        return (mean_feat.float() + delta).to(dtype=input_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, seq_len={self.seq_len}, "
            f"n_measurements={self.n_measurements}, bypass={self.bypass_quantum}"
        )
