"""
quantum_models/quantum_gated_adapter.py

Input-adaptive gated quantum adapter (Research Q2 — KIT program).

Same residual formulation as QuantumAdapter, but with a per-sample scalar gate
that controls how much of the quantum correction to apply:

    g(x)  = sigmoid(gate_net(x))   ∈ (0, 1)   — input-adaptive gate
    delta  = upscale(VQC(pre_net(x)))           — quantum (or classical) correction
    output = x + g * delta                      — gated residual

If the model learns g → 0 for all inputs: VQC correction is suppressed everywhere
  → publishable finding: "optimal quantum weight is zero across all inputs"
If g > 0 for some inputs: those tracklets benefit from quantum correction
  → answer to KIT Q2: which inputs prefer quantum?

bypass_quantum=True: delta from classical Linear+ReLU instead of VQC (gate still present).
  Useful ablation: does the gate collapse to 0 because of quantum specifically,
  or because any correction is unwanted?

Gate init: bias=0 → g=0.5 at init (neutral — equal classical/quantum weight).
Gate analysis: call forward(x, return_gates=True) to get (output, g_values) for logging.
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class GatedQuantumAdapter(nn.Module):
    """
    Input-adaptive gated quantum residual adapter.

    output = x + sigmoid(gate_net(x)) * upscale(VQC(pre_net(x)))

    Args:
        in_features  (int): Feature dimension (e.g. 768).
        n_qubits     (int): Number of qubits. Default 8.
        n_layers     (int): StronglyEntanglingLayers depth. Default 2.
        bypass_quantum (bool): Replace VQC with Linear+ReLU for ablation.
        device_name  (str): PennyLane device. Default 'default.qubit'.
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_quantum_features = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        # Gate: scalar per sample. bias=0 → g=0.5 at init (neutral).
        self.gate_net = nn.Linear(in_features, 1, bias=True)

        # Pre-net: compress in_features → n_qubits angles.
        self.pre_net = nn.Linear(in_features, n_qubits, bias=False)

        if bypass_quantum:
            self.classical_expansion = nn.Sequential(
                nn.Linear(n_qubits, self.n_quantum_features, bias=False),
                nn.ReLU(),
            )
        else:
            dev = qml.device(device_name, wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return qml.probs(wires=range(n_qubits))

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            self.qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

        # Upscale: 2^n_qubits → in_features. Near-zero init → delta ≈ 0 at init.
        self.upscale = nn.Linear(self.n_quantum_features, in_features, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        nn.init.zeros_(self.gate_net.bias)         # g = 0.5 at init
        nn.init.kaiming_normal_(self.gate_net.weight, a=0, mode="fan_in")
        if self.bypass_quantum:
            nn.init.kaiming_normal_(self.classical_expansion[0].weight, a=0, mode="fan_in")
        else:
            nn.init.normal_(self.qlayer.weights, mean=0, std=0.01)
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)

    def forward(
        self, x: torch.Tensor, return_gates: bool = False
    ):
        """
        Args:
            x            : [B, in_features]
            return_gates : If True, return (output, g) where g is [B] gate values.
        Returns:
            x_adapted [B, in_features], or (x_adapted, g [B]) if return_gates=True.
        """
        input_dtype  = x.dtype
        x_f = x.float()

        # Gate: [B, 1] → broadcast over in_features.
        g = torch.sigmoid(self.gate_net(x_f))   # [B, 1]

        # Correction branch.
        angles = torch.sigmoid(self.pre_net(x_f)) * math.pi   # [B, n_qubits]
        if self.bypass_quantum:
            q_feat = self.classical_expansion(angles)
        else:
            q_feat = self.qlayer(angles.float())
        delta = self.upscale(q_feat)   # [B, in_features]

        # Gated residual.
        x_adapted = (x_f + g * delta).to(dtype=input_dtype)

        if return_gates:
            return x_adapted, g.squeeze(1).detach().cpu()
        return x_adapted

    def extra_repr(self) -> str:
        mode = "classical_bypass" if self.bypass_quantum else "VQC"
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, mode={mode}"
        )
