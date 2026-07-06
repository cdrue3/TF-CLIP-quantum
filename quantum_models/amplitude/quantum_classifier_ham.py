"""
quantum_models/amplitude/quantum_classifier_ham.py

Hamiltonian-encoded quantum classifier head.

Same interface as QuantumClassifier (drop-in replacement for nn.Linear classifier heads)
but uses Hamiltonian encoding instead of angle encoding:

    x [B, in_features]
    → H(x) = Σᵢ xᵢ Pᵢ  (Pauli basis expansion, no compression)
    → U = e^{-iH}
    → |ψ⟩ = U|0⟩  (first column of unitary)
    → StatePrep → StronglyEntanglingLayers(trainable) → probs [B, 2^n_qubits]
    → post_net: Linear(2^n_qubits → num_classes)

No pre_net compression — features are encoded directly as Hamiltonian coefficients.
Requires 4^n_qubits - 1 >= in_features (n_qubits=5 gives 1023 >= 768).
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import math
from itertools import product as iproduct


def _build_pauli_basis(n_qubits: int, n_paulis: int) -> torch.Tensor:
    I2 = np.eye(2, dtype=complex)
    X  = np.array([[0, 1], [1, 0]], dtype=complex)
    Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z  = np.array([[1, 0], [0, -1]], dtype=complex)
    pm = {0: I2, 1: X, 2: Y, 3: Z}

    mats = []
    for ops in iproduct(range(4), repeat=n_qubits):
        if all(o == 0 for o in ops):
            continue
        mat = pm[ops[0]]
        for o in ops[1:]:
            mat = np.kron(mat, pm[o])
        mats.append(mat)
        if len(mats) == n_paulis:
            break

    return torch.tensor(np.stack(mats), dtype=torch.complex64)


class QuantumClassifierHam(nn.Module):
    """
    Hamiltonian-encoded quantum classifier.

    Args:
        in_features    (int): Input feature dimension (768 or 512).
        num_classes    (int): Number of output identity classes.
        n_qubits       (int): Number of qubits. Default 5 (4^5-1=1023 >= 768).
        n_layers       (int): StronglyEntanglingLayers depth. Default 2.
        bypass_quantum (bool): If True, replace VQC with Linear+ReLU.
        device_name    (str): PennyLane device.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        n_qubits: int = 5,
        n_layers: int = 2,
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
        encoding: str = "hamiltonian",
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_measurements = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum
        self.encoding = encoding

        assert 4 ** n_qubits - 1 >= in_features, (
            f"n_qubits={n_qubits} gives only {4**n_qubits - 1} Paulis < in_features={in_features}")

        self.register_buffer('pauli_basis', _build_pauli_basis(n_qubits, in_features))

        if bypass_quantum:
            self.classical_expansion = nn.Sequential(
                nn.Linear(in_features, self.n_measurements, bias=False),
                nn.ReLU(),
            )
        else:
            n_q = n_qubits
            dev = qml.device(device_name, wires=n_q)

            @qml.qnode(dev, interface='torch', diff_method='backprop')
            def _circuit(state, weights):
                qml.StatePrep(state, wires=range(n_q))
                qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                return qml.probs(wires=range(n_q))

            self.circuit = _circuit
            weight_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=n_layers, n_wires=n_q
            )
            self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))

        self.post_net = nn.Linear(self.n_measurements, num_classes, bias=False)
        self._init_weights()

    def _init_weights(self):
        if self.bypass_quantum:
            nn.init.kaiming_normal_(self.classical_expansion[0].weight, a=0, mode="fan_in")
        else:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.01)
        nn.init.kaiming_uniform_(self.post_net.weight, a=math.sqrt(5))

    def _apply(self, fn):
        super()._apply(fn)
        if not self.bypass_quantum:
            self.qlayer_weights.data = self.qlayer_weights.data.cpu().float()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]
        Returns:
            logits: [B, num_classes]
        """
        input_dtype = x.dtype
        input_device = x.device

        x = x.float()

        if self.bypass_quantum:
            x = self.classical_expansion(x)
        else:
            B = x.shape[0]

            H = torch.einsum('bi,ijk->bjk',
                             x.cpu().to(torch.complex64),
                             self.pauli_basis.cpu())

            U = torch.matrix_exp(-1j * H)
            initial_state = U[:, :, 0].contiguous()

            norms = initial_state.norm(dim=1, keepdim=True).clamp(min=1e-8)
            initial_state = initial_state / norms

            x = self.circuit(
                initial_state,
                self.qlayer_weights.cpu().float()
            ).float().to(input_device)

        x = self.post_net(x)
        return x.to(input_dtype)

    def extra_repr(self) -> str:
        mode = "classical_ablation" if self.bypass_quantum else f"n_layers={self.n_layers}"
        return (
            f"in_features={self.in_features}, num_classes={self.num_classes}, "
            f"n_qubits={self.n_qubits}, encoding=hamiltonian, {mode}, "
            f"n_measurements={self.n_measurements}"
        )
