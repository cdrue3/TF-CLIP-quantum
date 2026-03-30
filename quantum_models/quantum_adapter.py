"""
quantum_models/quantum_adapter.py

VQC Adapter — shared residual quantum feature transformation.

Survey §3.2.2 (Feature Extraction level), §3.3.1 (Sequential/Residual pattern).

Architecture:
    Instead of per-head VQC (4 × VQC evals in qclassifier/qfeatext), a SINGLE
    shared QuantumAdapter is applied to the primary backbone feature BEFORE the
    classical classifier head. All other heads remain pure nn.Linear.

    CLIP(768) → QuantumAdapter(residual) → BN_neck → nn.Linear(625)   [primary path]
                                         (remaining 3 heads: classical, unadapted)

QuantumAdapter residual formulation:
    x [B, in_features]
    ├───────────────────────────────────────────────────────────► identity (skip)
    └──► pre_net : Linear(in_features → n_qubits, bias=False)
      → sigmoid(x) * π                                              (0, π)
      → qlayer  : AngleEmbedding + StronglyEntanglingLayers → probs()
      → upscale : Linear(2^n_qubits → in_features, bias=False)
                                                                   ▼
    x_adapted = x + upscale(qlayer(sigmoid(pre_net(x)) * π))
    [B, in_features] — same shape, same dtype, same device as input

Why different from previous experiments:
    qclassifier (sequential): 4 × VQC, each replaces a classifier head.
        x → pre_net(n_q) → VQC → probs(2^n_q) → post_net → 625
    qfeatext (parallel): 4 × VQC, each augments its head's input by concatenation.
        [x || VQC(x)] → post_net → 625
    adapter (this file): 1 × VQC, shared, residual on primary path.
        x → VQC_residual → x_adapted → nn.Linear(768→625)

Key properties:
    - Residual init (upscale std=0.001): adapter starts near-identity; classical
      features are preserved at init, adapter learns incrementally.
    - One VQC run per forward pass (vs 4 in prior approaches).
    - Classifier heads remain classical nn.Linear — full gradient flow.
    - Cannot degrade classification at init; can only add signal.

Install:
    pip install pennylane==0.33.1
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumAdapter(nn.Module):
    """
    Residual VQC adapter: x_adapted = x + upscale(VQC(pre_net(x))).

    Applies a variational quantum circuit as a residual connection.
    Output shape is identical to input — drop-in residual for any 768-dim feature.

    Args:
        in_features  (int): Feature dimension to adapt (e.g. 768 for ViT-B-16).
        n_qubits     (int): Number of qubits. Default 8 → 256 quantum features.
        n_layers     (int): StronglyEntanglingLayers depth. Default 2.
        device_name  (str): PennyLane device. Default 'default.qubit' (CPU sim).
        encoding     (str): 'angle' (default) or 'dense_angle'.
                            'angle':       pre_net: 768→n_qubits; AngleEmbedding RY.
                            'dense_angle': pre_net: 768→2*n_qubits; per-qubit RY(angle)+RZ(phase).
                                           Doubles information per qubit (survey Table 3).
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        device_name: str = "default.qubit",
        bypass_quantum: bool = False,
        encoding: str = "angle",
    ):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_quantum_features = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum
        self.encoding = encoding

        # For dense_angle, pre_net outputs 2*n_qubits (angles + phases).
        pre_out = 2 * n_qubits if encoding == "dense_angle" else n_qubits
        self.pre_net = nn.Linear(in_features, pre_out, bias=False)

        if bypass_quantum:
            # Classical ablation: Linear(pre_out→2^n_qubits)+ReLU.
            self.classical_expansion = nn.Sequential(
                nn.Linear(pre_out, self.n_quantum_features, bias=False),
                nn.ReLU(),
            )
        else:
            dev = qml.device(device_name, wires=n_qubits)

            if encoding == "dense_angle":
                # Dense angle: RY(angle_j * π) + RZ(phase_j * 2π) per qubit.
                # Encodes 2 features per qubit (doubles info vs standard angle).
                # inputs: [B, 2*n_qubits], first n_qubits = angles, last n_qubits = phases.
                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def _circuit(inputs, weights):
                    for j in range(n_qubits):
                        qml.RY(inputs[..., j] * math.pi, wires=j)
                        qml.RZ(inputs[..., n_qubits + j] * 2 * math.pi, wires=j)
                    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                    return qml.probs(wires=range(n_qubits))
            else:
                # Standard angle: AngleEmbedding RY(x * π) per qubit.
                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def _circuit(inputs, weights):
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                    return qml.probs(wires=range(n_qubits))

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            self.qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

        # Upscale: projects 2^n_qubits features back to in_features.
        # Near-zero init ensures adapter starts as identity (residual delta ≈ 0).
        self.upscale = nn.Linear(self.n_quantum_features, in_features, bias=False)

        self._init_weights()

    def _init_weights(self):
        # pre_net: fan_in kaiming — output std ≈ sqrt(2/in_features), sigmoid near π/2.
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        if self.bypass_quantum:
            # classical_expansion Linear: kaiming_normal (fan_in).
            nn.init.kaiming_normal_(self.classical_expansion[0].weight, a=0, mode="fan_in")
        else:
            # qlayer: near-identity init to avoid barren plateau at init.
            nn.init.normal_(self.qlayer.weights, mean=0, std=0.01)
        # upscale: near-zero so residual delta ≈ 0 at init (identity adapter).
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)

    def to(self, *args, **kwargs):
        """Pin qlayer to CPU; pre_net and upscale can live on GPU."""
        super().to(*args, **kwargs)
        if not self.bypass_quantum:
            self.qlayer.to(device=torch.device("cpu"), dtype=torch.float32)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]  (float32 or float16, any device)
        Returns:
            x_adapted: [B, in_features]  (same shape/dtype/device as input)
        """
        input_dtype  = x.dtype
        input_device = x.device

        x_f = x.float()
        raw = self.pre_net(x_f)   # [B, n_qubits] or [B, 2*n_qubits]

        if self.encoding == "dense_angle":
            # Split into angle (first n_qubits) and phase (last n_qubits).
            # Angles ∈ (0, π); phases ∈ (0, 2π).
            angles = torch.sigmoid(raw[..., :self.n_qubits]) * math.pi
            phases = torch.sigmoid(raw[..., self.n_qubits:]) * 2 * math.pi
            enc = torch.cat([angles, phases], dim=-1)   # [B, 2*n_qubits]
        else:
            enc = torch.sigmoid(raw) * math.pi          # [B, n_qubits]

        if self.bypass_quantum:
            q_feat = self.classical_expansion(enc)             # [B, 2^n_qubits]
        else:
            enc_cpu = enc.cpu().float()
            q_feat = self.qlayer(enc_cpu).to(input_device)    # [B, 2^n_qubits]

        delta = self.upscale(q_feat)                           # [B, in_features]
        x_adapted = x_f + delta

        return x_adapted.to(input_dtype)

    def extra_repr(self) -> str:
        mode = "classical_bypass" if self.bypass_quantum else "VQC"
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, encoding={self.encoding}, mode={mode}"
        )
