import math
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from itertools import product as iproduct


def _build_pauli_basis(n_qubits: int, n_paulis: int) -> torch.Tensor:
    """
    Build a fixed basis of n_paulis Pauli operators on n_qubits qubits.

    A Pauli operator on n qubits is a tensor product of single-qubit Paulis,
    e.g. X⊗Z⊗I for n=3. There are 4^n total (including all-identity), so
    4^n - 1 non-identity operators. For n=5: 1023 operators, which covers
    our 768-dimensional feature space with room to spare.

    Operators are enumerated in lexicographic order over {I,X,Y,Z}^n,
    skipping the all-identity string. Ordering is arbitrary for our use —
    what matters is that feature index i always maps to the same Pauli Pᵢ.

    Returns: [n_paulis, 2^n_qubits, 2^n_qubits] complex64
             Each slice [i] is the 2^n × 2^n matrix of the i-th Pauli operator.
    """
    I2 = np.eye(2, dtype=complex)
    X  = np.array([[0, 1], [1, 0]], dtype=complex)
    Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z  = np.array([[1, 0], [0, -1]], dtype=complex)
    pm = {0: I2, 1: X, 2: Y, 3: Z}

    mats = []
    for ops in iproduct(range(4), repeat=n_qubits):
        if all(o == 0 for o in ops):
            continue  # skip IIIII — identity carries no information
        # Build n-qubit matrix as Kronecker product of single-qubit Paulis
        mat = pm[ops[0]]
        for o in ops[1:]:
            mat = np.kron(mat, pm[o])
        mats.append(mat)
        if len(mats) == n_paulis:
            break

    return torch.tensor(np.stack(mats), dtype=torch.complex64)


class QTDHam(nn.Module):
    """
    QTD with Hamiltonian Encoding — compression-free via Pauli basis expansion.

    Each frame difference x ∈ R^in_features is encoded as the coefficients of a
    Hamiltonian H(x) = Σᵢ xᵢ Pᵢ over n_qubits-qubit Pauli operators. The initial
    quantum state is U|0⟩ where U = e^{-iH(x)} computed via torch.matrix_exp.

    For in_features=768 features and n_qubits=5: 4^5-1=1023 available Paulis ≥ 768.
    State space: 2^5=32 dimensions. No pre_net compression.

    Pauli basis computed at init and stored as a buffer (not trained).
    """

    def __init__(self, in_features=768, n_qubits=6, n_layers=2, seq_len=8,
                 bypass_quantum=False, device_name='default.qubit'):
        super().__init__()
        assert 4 ** n_qubits - 1 >= in_features, (
            f"n_qubits={n_qubits} gives only {4**n_qubits - 1} Paulis < in_features={in_features}")

        self.in_features    = in_features
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.seq_len        = seq_len
        self.n_diffs        = seq_len - 1
        self.n_states       = 2 ** n_qubits   # 64 for n_qubits=6
        self.bypass_quantum = bypass_quantum

        # Pauli basis: [in_features, n_states, n_states] complex64 — non-trainable.
        # Computed once at init; each of the 768 feature dimensions is permanently
        # assigned to one Pauli operator. This is the encoding dictionary.
        self.register_buffer('pauli_basis', _build_pauli_basis(n_qubits, in_features))

        if not bypass_quantum:
            n_q = n_qubits
            dev = qml.device(device_name, wires=n_q)

            @qml.qnode(dev, interface='torch', diff_method='backprop')
            def _circuit(state, weights):
                # ---- THIS IS WHERE THE ENCODING ENTERS THE QUANTUM CIRCUIT ----
                # state: [B, 2^n_q] complex — the Hamiltonian-evolved initial state U|0>.
                # StatePrep loads this as the full quantum state, preserving all
                # complex phases that encode the input features.
                # weights: [n_layers, n_q, 3] — the only TRAINABLE parameters.
                qml.StatePrep(state, wires=range(n_q))
                # Trainable entangling layers act on the encoded state.
                # These learn to rotate/entangle the Hamiltonian-encoded information
                # into a measurement outcome that's useful for re-ID.
                qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                # Measure outcome probabilities: [B, 2^n_q] = [B, 32]
                return qml.probs(wires=range(n_q))

            self.circuit = _circuit
            weight_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=n_layers, n_wires=n_q
            )
            # Only these weights are trained — the encoding itself has no learned params
            self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))

        # Maps circuit outputs to in_features — no skip, circuit output is sole signal
        self.upscale = nn.Linear(self.n_states, in_features, bias=False)
        self._init_weights()

    def _init_weights(self):
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.01)
        nn.init.normal_(self.upscale.weight, mean=0, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, in_features] → [B, n_states] — circuit output IS the descriptor"""
        if self.bypass_quantum:
            return x.mean(1)

        input_dtype  = x.dtype
        B, T, D = x.shape

        # Motion signal: T-1 consecutive frame differences, each 768-dim
        diffs      = x[:, 1:] - x[:, :-1]                          # [B, T-1, D]
        diffs_flat = diffs.float().reshape(B * self.n_diffs, D)     # [B*(T-1), D]

        # =====================================================================
        # HAMILTONIAN ENCODING — this is where features become a quantum state
        # =====================================================================
        # Step 1: Build H(x) = Σᵢ xᵢ Pᵢ
        #   Each feature value xᵢ scales its assigned Pauli operator Pᵢ.
        #   The sum is a 32×32 Hermitian matrix — a valid quantum Hamiltonian.
        #   einsum: contracts feature dim (i) against Pauli index (i) in the basis,
        #   producing a batch of Hamiltonians [B*(T-1), 32, 32].
        H = torch.einsum('bi,ijk->bjk',
                         diffs_flat.to(torch.complex64),
                         self.pauli_basis)                          # [B*(T-1), 32, 32]

        # Step 2: Compute unitary U = e^{-iH}
        #   Because H is Hermitian (real coefficients × Hermitian Paulis = Hermitian),
        #   e^{-iH} is guaranteed unitary. This is the quantum time-evolution operator.
        #   The -i factor gives it the standard quantum mechanics form e^{-iHt} with t=1.
        U = torch.matrix_exp(-1j * H)                               # [B*(T-1), 32, 32]

        # Step 3: Initial state = U|0⟩ — apply unitary to the ground state
        #   |0⟩ in the computational basis is [1, 0, 0, ..., 0] (32-dim for 5 qubits).
        #   U|0⟩ is simply the first column of U.
        #   This state encodes all 768 input features via complex amplitudes and phases.
        initial_state = U[:, :, 0].contiguous()                     # [B*(T-1), 32] complex

        # Re-normalise: matrix_exp accumulates ~1e-5 floating point error
        norms = initial_state.norm(dim=1, keepdim=True).clamp(min=1e-8)
        initial_state = initial_state / norms
        # =====================================================================

        # Run the PennyLane circuit: StatePrep loads initial_state, then
        # StronglyEntanglingLayers (trainable) transforms it, then we measure probs
        q_out = self.circuit(
            initial_state,
            self.qlayer_weights.float()
        ).float()                                                    # [B*(T-1), 32]

        # Average over the T-1 differences — each contributes one measurement
        q_out = q_out.reshape(B, self.n_diffs, self.n_states).mean(1)  # [B, n_states]

        return self.upscale(q_out).to(dtype=input_dtype)

    def extra_repr(self):
        return (f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
                f"n_states={self.n_states}, n_layers={self.n_layers}, "
                f"n_diffs={self.n_diffs}, bypass={self.bypass_quantum}")
