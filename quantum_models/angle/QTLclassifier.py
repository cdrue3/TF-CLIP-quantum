"""
quantum_models/QTLclassifier.py

Proper Quantum Transfer Learning (QTL) classifier module.

Based on: Mari et al., "Transfer learning in hybrid classical-quantum
neural networks", Quantum 4, 340 (2020).

Structural differences from QuantumClassifier (quantum_layers.py):
    1. dress_layer: Linear(in_features, n_qubits, bias=True) — the QTL
       "dress" layer per Mari et al. nomenclature.
    2. Angle scaling: tanh(dress_layer(x)) * π — covers the full angular
       range (−π, π) as in the original QTL paper.
    3. Single head: one VQC per model. The qclassifier had 4 redundant
       heads over the same task — proper QTL uses one.
    4. Designed for a FROZEN classical backbone. The module receives
       pre-computed backbone features; no joint training with the encoder.
    5. output_layer init: std=0.001 so the VQC shapes its probability
       distribution before the output layer imposes strong gradients.

Why this is proper QTL:
    Classical transfer learning fine-tunes a frozen backbone's head on a
    new task. Quantum TL replaces that fine-tuning step with a VQC: the
    frozen backbone is a fixed feature extractor, and the VQC is the sole
    learner. The key properties are:
        - Classical encoder: frozen, no gradient flow
        - Quantum head: the only trainable component
        - Single VQC decision pathway (not 4 duplicates)

Why tanh instead of sigmoid:
    Mari et al. use tanh(·) * π to map dress outputs to (−π, π), covering
    the full Bloch sphere meridian.  sigmoid(·) * π only covers (0, π).
    At dress_layer output ≈ 0, tanh'(0) = 1 (maximum gradient), matching
    the sigmoid advantage from QuantumClassifier but with full range.
"""

import math
import torch
import torch.nn as nn
import pennylane as qml


class QTLClassifier(nn.Module):
    """
    Quantum Transfer Learning classifier (Mari et al., 2020).

    Args:
        in_features  (int): Dimension of frozen backbone features (e.g. 768).
        num_classes  (int): Number of output identity classes.
        n_qubits     (int): Number of qubits. Default 8.
        n_layers     (int): Number of variational entangling layers. Default 2.
        device_name  (str): PennyLane device string. Default "default.qubit".
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features    = in_features
        self.num_classes    = num_classes
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.n_measurements = 2 ** n_qubits

        # ------------------------------------------------------------------
        # 1. Dress layer (Mari et al. nomenclature)
        #    Learnable projection: in_features → n_qubits.
        #    bias=True per the original QTL paper.
        #    Angle mapping: tanh(dress_layer(x)) * π → (−π, π).
        # ------------------------------------------------------------------
        self.dress_layer = nn.Linear(in_features, n_qubits, bias=True)

        # ------------------------------------------------------------------
        # 2. Variational quantum circuit
        #    AngleEmbedding (RY) + StronglyEntanglingLayers + qml.probs().
        #    Same ansatz as QuantumClassifier for fair comparison.
        #    Circuit runs on CPU (PennyLane default.qubit state-vector).
        # ------------------------------------------------------------------
        dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

        self.qlayer = qml.qnn.TorchLayer(_circuit, {"weights": (n_layers, n_qubits, 3)})

        # ------------------------------------------------------------------
        # 3. Output layer
        #    Maps VQC probability vector (2^n_qubits) → num_classes logits.
        # ------------------------------------------------------------------
        self.output_layer = nn.Linear(self.n_measurements, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        # dress_layer: fan_in Kaiming — output std ≈ 1 so tanh operates in
        # its near-linear regime (tanh'(0) = 1, maximum gradient at init).
        nn.init.kaiming_normal_(self.dress_layer.weight, a=0, mode="fan_in")
        nn.init.zeros_(self.dress_layer.bias)

        # qlayer: near-identity init — avoids barren plateau at random weights.
        nn.init.normal_(self.qlayer.weights, mean=0.0, std=0.01)

        # output_layer: small init — gives VQC time to shape its probability
        # distribution before the output layer dominates the gradient signal.
        nn.init.normal_(self.output_layer.weight, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features] — frozen backbone features (any dtype/device).
        Returns:
            logits: [B, num_classes]
        """
        input_dtype  = x.dtype
        x = x.float()

        # Stage 1: dress layer + tanh·π → angles in (−π, π)
        x = torch.tanh(self.dress_layer(x)) * math.pi   # [B, n_qubits]

        # Stage 2: VQC
        x = self.qlayer(x.float())                       # [B, 2^n_qubits]

        # Stage 3: output projection
        x = self.output_layer(x)                         # [B, num_classes]

        return x.to(dtype=input_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, num_classes={self.num_classes}, "
            f"n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"n_measurements={self.n_measurements}, angle=tanh*π"
        )
