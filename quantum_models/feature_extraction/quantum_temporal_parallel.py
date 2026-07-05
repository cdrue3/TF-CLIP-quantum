"""
Parallel Quantum-Classical Temporal Aggregation + SQP + BER.

Two independent paths process the same [B, T, 768] tracklet features:
  Classical: mean_pool → [B, 768]
  Quantum:   VQC temporal re-uploading → [B, 768]

Fusion modes:
  'concat': concat [B, 1536] → Linear+ReLU → [B, 768]
  'gated':  g·classical + (1-g)·quantum, where g = sigmoid(Linear(classical))

Quantum branch options:
  angle encoding (default), dense angle, or hamiltonian.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from itertools import product as iproduct


def _build_pauli_basis(n_qubits, n_paulis):
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


class QuantumTemporalParallel(nn.Module):

    def __init__(self, in_features=768, n_qubits=8, n_layers=2, seq_len=8,
                 bypass_quantum=False, dense_encoding=False,
                 hamiltonian=False, fusion_mode='concat',
                 device_name='default.qubit'):
        super().__init__()
        self.in_features    = in_features
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.seq_len        = seq_len
        self.bypass_quantum = bypass_quantum
        self.dense_encoding = dense_encoding
        self.hamiltonian    = hamiltonian
        self.fusion_mode    = fusion_mode

        self._noise_scale = 0.0
        self._last_probs  = None
        self.last_gates   = None  # for gated fusion analysis

        if hamiltonian:
            ham_n_q = n_qubits if 4**n_qubits - 1 >= in_features else 5
            assert 4**ham_n_q - 1 >= in_features
            self.ham_n_qubits = ham_n_q
            self.n_states = 2 ** ham_n_q
            self.register_buffer('pauli_basis', _build_pauli_basis(ham_n_q, in_features))

            if not bypass_quantum:
                dev = qml.device(device_name, wires=ham_n_q)

                @qml.qnode(dev, interface='torch', diff_method='backprop')
                def _circuit(state, weights):
                    qml.StatePrep(state, wires=range(ham_n_q))
                    qml.StronglyEntanglingLayers(weights, wires=range(ham_n_q))
                    return qml.probs(wires=range(ham_n_q))

                self.circuit = _circuit
                weight_shape = qml.StronglyEntanglingLayers.shape(
                    n_layers=n_layers, n_wires=ham_n_q)
                self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))

            self.q_upscale = nn.Linear(self.n_states, in_features, bias=False)
        else:
            self.n_measurements = 2 ** n_qubits
            pre_net_out = 2 * n_qubits if dense_encoding else n_qubits
            self.pre_net = nn.Linear(in_features, pre_net_out, bias=False)

            if not bypass_quantum:
                n_q = n_qubits
                dev = qml.device(device_name, wires=n_q)

                if dense_encoding:
                    @qml.qnode(dev, interface="torch", diff_method="backprop")
                    def _circuit(angles_2d, weights):
                        for t in range(seq_len):
                            qml.AngleEmbedding(angles_2d[t, :, :n_q], wires=range(n_q), rotation="Y")
                            qml.AngleEmbedding(angles_2d[t, :, n_q:], wires=range(n_q), rotation="Z")
                            qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                        return qml.probs(wires=range(n_q))
                else:
                    @qml.qnode(dev, interface="torch", diff_method="backprop")
                    def _circuit(angles_2d, weights):
                        for t in range(seq_len):
                            qml.AngleEmbedding(angles_2d[t], wires=range(n_q), rotation="Y")
                            qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                        return qml.probs(wires=range(n_q))

                self.circuit = _circuit
                weight_shape = qml.StronglyEntanglingLayers.shape(
                    n_layers=n_layers, n_wires=n_q)
                self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))

            self.q_upscale = nn.Linear(self.n_measurements, in_features, bias=False)

        # Fusion layer
        if fusion_mode == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(in_features * 2, in_features, bias=False),
                nn.ReLU(),
            )
        elif fusion_mode == 'gated':
            self.gate_net = nn.Linear(in_features, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        if not self.hamiltonian:
            nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.005)
        nn.init.normal_(self.q_upscale.weight, mean=0, std=0.001)
        if self.fusion_mode == 'concat':
            nn.init.eye_(self.fusion[0].weight[:, :self.in_features])
            nn.init.zeros_(self.fusion[0].weight[:, self.in_features:])
        elif self.fusion_mode == 'gated':
            nn.init.normal_(self.gate_net.weight, mean=0, std=0.01)
            nn.init.constant_(self.gate_net.bias, 2.0)  # sigmoid(2)≈0.88 → starts mostly classical

    def _quantum_forward_angle(self, x, B, T, D):
        pre_net_dim = 2 * self.n_qubits if self.dense_encoding else self.n_qubits
        angles = torch.sigmoid(self.pre_net(x.float().reshape(B * T, D))) * math.pi
        angles = angles.reshape(B, T, pre_net_dim)
        angles_f = angles.permute(1, 0, 2).float()
        weights_f = self.qlayer_weights.float()
        if self.training and self._noise_scale > 0.0:
            weights_f = weights_f + self._noise_scale * torch.randn_like(weights_f)
        q_out = self.circuit(angles_f, weights_f).float()
        return q_out

    def _quantum_forward_ham(self, x, B, T, D):
        frames_flat = x.float().reshape(B * T, D)
        H = torch.einsum('bi,ijk->bjk',
                         frames_flat.to(torch.complex64),
                         self.pauli_basis)
        U = torch.matrix_exp(-1j * H)
        initial_state = U[:, :, 0].contiguous()
        norms = initial_state.norm(dim=1, keepdim=True).clamp(min=1e-8)
        initial_state = initial_state / norms

        weights_f = self.qlayer_weights.float()
        if self.training and self._noise_scale > 0.0:
            weights_f = weights_f + self._noise_scale * torch.randn_like(weights_f)

        q_out = self.circuit(initial_state, weights_f).float()
        q_out = q_out.reshape(B, T, self.n_states).mean(1)
        return q_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        B, T, D = x.shape

        classical_feat = x.mean(1).float()

        if self.bypass_quantum:
            return classical_feat.to(dtype=input_dtype)

        if self.hamiltonian:
            q_out = self._quantum_forward_ham(x, B, T, D)
        else:
            q_out = self._quantum_forward_angle(x, B, T, D)

        self._last_probs = q_out.detach()
        quantum_feat = self.q_upscale(q_out)

        if self.fusion_mode == 'concat':
            fused = self.fusion(torch.cat([classical_feat, quantum_feat], dim=1))
        elif self.fusion_mode == 'gated':
            g = torch.sigmoid(self.gate_net(classical_feat))
            self.last_gates = g.squeeze(1).detach().cpu()
            fused = g * classical_feat + (1 - g) * quantum_feat

        return fused.to(dtype=input_dtype)

    def extra_repr(self):
        if self.hamiltonian:
            enc = f"hamiltonian (n_q={self.ham_n_qubits}, {self.n_states} states)"
        elif self.dense_encoding:
            enc = "dense_angle"
        else:
            enc = "angle"
        return (f"PARALLEL: classical || quantum, encoding={enc}, "
                f"fusion={self.fusion_mode}, SQP+BER, bypass={self.bypass_quantum}")
