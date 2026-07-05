"""
Temporal Quantum Aggregation with Hamiltonian Encoding + SQP + BER.

Encodes ALL 768 features per frame via Pauli basis expansion — no pre_net
compression bottleneck. Each frame x ∈ R^768 maps to a Hamiltonian
H(x) = Σᵢ xᵢ Pᵢ where Pᵢ are n-qubit Pauli operators. The quantum state
U|0⟩ = e^{-iH(x)}|0⟩ preserves all feature information as complex amplitudes.

n_qubits=5: 4^5-1 = 1023 Paulis ≥ 768 features, state dim = 2^5 = 32
n_qubits=6: 4^6-1 = 4095 Paulis ≥ 768 features, state dim = 2^6 = 64
"""

import math
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
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


class QuantumTemporalHam(nn.Module):
    """
    TQA with Hamiltonian encoding + SQP noise + BER entropy exposure.

    Each of T frames is Hamiltonian-encoded into a full quantum state,
    processed by trainable StronglyEntanglingLayers, measured, then
    averaged. Skip connection: output = mean_pool(x) + upscale(avg_probs).
    """

    def __init__(self, in_features=768, n_qubits=5, n_layers=2, seq_len=8,
                 bypass_quantum=False, device_name='default.qubit'):
        super().__init__()
        assert 4 ** n_qubits - 1 >= in_features, (
            f"n_qubits={n_qubits} gives only {4**n_qubits - 1} Paulis < {in_features}")

        self.in_features    = in_features
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.seq_len        = seq_len
        self.n_states       = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        self._noise_scale = 0.0
        self._last_probs  = None

        self.register_buffer('pauli_basis', _build_pauli_basis(n_qubits, in_features))

        if not bypass_quantum:
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

        self.upscale = nn.Linear(self.n_states, in_features, bias=False)
        self._init_weights()

    def _init_weights(self):
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.005)
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_feat = x.mean(1)
        if self.bypass_quantum:
            return mean_feat

        input_dtype = x.dtype
        B, T, D = x.shape

        frames_flat = x.float().reshape(B * T, D)

        # H(x) = Σᵢ xᵢ Pᵢ  →  [B*T, n_states, n_states] complex
        H = torch.einsum('bi,ijk->bjk',
                         frames_flat.to(torch.complex64),
                         self.pauli_basis)

        # U = e^{-iH}, state = U|0⟩ = first column of U
        U = torch.matrix_exp(-1j * H)
        initial_state = U[:, :, 0].contiguous()

        # Renormalise
        norms = initial_state.norm(dim=1, keepdim=True).clamp(min=1e-8)
        initial_state = initial_state / norms

        weights_f = self.qlayer_weights.float()
        if self.training and self._noise_scale > 0.0:
            weights_f = weights_f + self._noise_scale * torch.randn_like(weights_f)

        q_out = self.circuit(initial_state, weights_f).float()

        # Average over T frames
        q_out = q_out.reshape(B, T, self.n_states).mean(1)

        self._last_probs = q_out.detach()

        delta = self.upscale(q_out)
        return (mean_feat.float() + delta).to(dtype=input_dtype)

    def extra_repr(self):
        return (f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
                f"n_states={self.n_states}, n_layers={self.n_layers}, "
                f"encoding=hamiltonian (768 features, no compression), "
                f"SQP+BER, bypass={self.bypass_quantum}")
