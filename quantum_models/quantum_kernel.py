"""
quantum_models/quantum_kernel.py

Quantum Fidelity Kernel for retrieval similarity.

Replaces Euclidean distance in the gallery ranking step with a quantum kernel
computed only at eval/inference time — no backprop through the circuit during
the main training loop (avoids the gradient bottleneck of previous VQC approaches).

Feature map: IQP-style (Instantaneous Quantum Polynomial)
    U(x): Hadamard ⊗ n  →  RZ(x_k) per qubit  →  IsingZZ(x_i · x_{i+1}) for adjacent pairs

The IsingZZ cross-terms create entanglement between adjacent qubit dimensions,
making this kernel non-factorisable (cannot be reduced to a product of 1-qubit terms).
This is genuinely quantum: the kernel has no simple closed-form classical expression.

Kernel:  K(x1, x2) = |⟨0| U†(x1) U(x2) |0⟩|²  =  P(|0...0⟩) from the overlap circuit.
    K = 1.0 when x1 = x2 (same quantum state).
    K < 1.0 otherwise (decreasing with feature difference).

Distance:  d(x1, x2) = 1 − K  ∈ [0, 1]

Trainable component: pre_net (Linear in_features → n_qubits).
    Trained in train_qkernel.py via BCE pair loss on extracted features.
    The IQP circuit itself has no learnable parameters.

Eval: distance_matrix() computes [M, N] distances for eval_func().
    top_k mode: quantum kernel only on Euclidean top-K per query (for large galleries).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


class QuantumKernel(nn.Module):
    """
    Quantum fidelity kernel with IQP feature map.

    Args:
        in_features  (int): Input feature dimension (e.g., 768).
        n_qubits     (int): Number of qubits. More qubits = richer kernel, slower sim.
        device_name  (str): PennyLane device. Default 'default.qubit'.
    """

    def __init__(
        self,
        in_features: int = 768,
        n_qubits: int = 4,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits

        # Only trainable component: compress in_features → n_qubits angles.
        self.pre_net = nn.Linear(in_features, n_qubits, bias=False)
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")

        n_q = n_qubits
        dev = qml.device(device_name, wires=n_q)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _kernel_circuit(x1_enc, x2_enc):
            """
            Compute K(x1, x2) = |⟨0| U†(x1) U(x2) |0⟩|².

            x1_enc, x2_enc: [n_q] angles in (0, π), on CPU.

            IQP feature map U(x):
              1. Hadamard on all qubits            (creates superposition)
              2. RZ(x_k) on qubit k               (encodes features as phases)
              3. IsingZZ(x_i * x_{i+1}) on [i, i+1]  (entangles adjacent qubits
                                                        via ZZ interactions — not
                                                        reducible to 1-qubit terms)

            U†(x) applies gates in reverse order with negated angles:
              1. IsingZZ(−x_i * x_{i+1}) pairs, reversed order
              2. RZ(−x_k) per qubit
              3. Hadamard on all qubits   (self-adjoint: H† = H)
            """
            # ── U(x2) ─────────────────────────────────────────────────────────
            for i in range(n_q):
                qml.Hadamard(wires=i)
            for i in range(n_q):
                qml.RZ(x2_enc[i], wires=i)
            for i in range(n_q - 1):
                qml.IsingZZ(x2_enc[i] * x2_enc[i + 1], wires=[i, i + 1])

            # ── U†(x1): reverse order, negate angles ──────────────────────────
            for i in reversed(range(n_q - 1)):
                qml.IsingZZ(-(x1_enc[i] * x1_enc[i + 1]), wires=[i, i + 1])
            for i in range(n_q):
                qml.RZ(-x1_enc[i], wires=i)
            for i in range(n_q):
                qml.Hadamard(wires=i)

            return qml.probs(wires=range(n_q))

        self._circuit = _kernel_circuit

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map features to angles in (0, π): sigmoid(pre_net(x)) * π."""
        return torch.sigmoid(self.pre_net(x.float())) * math.pi

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Kernel values for a batch of pairs. Used for BCE training of pre_net.

        Gradients flow: loss → k_vals → circuit → x_enc.cpu() → x_enc → pre_net.weight.

        Args:
            x1, x2: [B, in_features]
        Returns:
            [B] kernel values ∈ (0, 1)
        """
        a_enc = self.encode(x1)        # [B, n_q], grad-connected to pre_net
        b_enc = self.encode(x2)

        a_cpu = a_enc.cpu()            # device transfer, grad still flows back
        b_cpu = b_enc.cpu()

        k_vals = [self._circuit(a_cpu[i], b_cpu[i])[0] for i in range(x1.shape[0])]
        return torch.stack(k_vals)     # [B]

    @torch.no_grad()
    def distance_matrix(
        self,
        qf: torch.Tensor,
        gf: torch.Tensor,
        top_k: int = None,
    ) -> np.ndarray:
        """
        Quantum kernel distance matrix for eval.

        Args:
            qf     : [M, in_features] — query features (CPU or GPU)
            gf     : [N, in_features] — gallery features
            top_k  : If set, use quantum kernel only for Euclidean top-K per query.
                     Non-top-K entries get classical Euclidean distance normalised to
                     [1, 2], so top-K items always rank before them.
                     Quantum distances ∈ [0, 1].
                     Set to None for a full M×N kernel matrix.
        Returns:
            [M, N] numpy float32 distance array (lower = more similar).
        """
        from utils.metrics import euclidean_distance

        self.pre_net.cpu()
        qf_cpu = qf.cpu().float()
        gf_cpu = gf.cpu().float()

        with torch.no_grad():
            qf_enc = self.encode(qf_cpu)   # [M, n_q]
            gf_enc = self.encode(gf_cpu)   # [N, n_q]

        M, N = qf_enc.shape[0], gf_enc.shape[0]

        if top_k is None:
            # Full matrix: all M×N pairs via circuit.
            dist = np.empty((M, N), dtype=np.float32)
            for i in range(M):
                if i % 10 == 0:
                    print(f"\r  Kernel matrix: query {i}/{M}", end="", flush=True)
                for j in range(N):
                    k = self._circuit(qf_enc[i], gf_enc[j])[0].item()
                    dist[i, j] = 1.0 - k
            print()
            return dist

        # Top-K reranking:
        # 1. Classical Euclidean for initial ranking (normalised to [1, 2] per query).
        # 2. Quantum kernel replaces distances for the top-K (values ∈ [0, 1]).
        # Result: top-K items always rank before non-top-K (no scale mismatch).
        eucl = euclidean_distance(qf.cpu(), gf.cpu())   # [M, N] numpy

        dist = np.empty((M, N), dtype=np.float32)
        for i in range(M):
            if i % 50 == 0:
                print(f"\r  Top-{top_k} reranking: query {i}/{M}", end="", flush=True)
            row = eucl[i]
            lo, hi = row.min(), row.max()
            dist[i] = (row - lo) / (hi - lo + 1e-8) + 1.0  # normalise to [1, 2]

            top_k_idx = np.argsort(row)[:top_k]
            for j in top_k_idx:
                k = self._circuit(qf_enc[i], gf_enc[j])[0].item()
                dist[i, j] = 1.0 - k   # replace with quantum distance ∈ [0, 1]

        print()
        return dist

    @torch.no_grad()
    def distance_matrix_blended(
        self,
        qf: torch.Tensor,
        gf: torch.Tensor,
        top_k: int,
        lambdas: list,
    ) -> dict:
        """
        Compute blended distance matrices for multiple λ values in a single circuit pass.

        For each (query i, gallery j) in Euclidean top-K:
            d(i,j) = (1-λ)*eucl_norm(i,j) + λ*(1 - K(i,j))
        For non-top-K entries:
            d(i,j) = eucl_norm(i,j) + 1.0   (always > any top-K entry which is ≤ 1.0)

        λ=0.0 → pure Euclidean ordering within top-K (same Rank-1 as Euclidean)
        λ=1.0 → pure quantum kernel (same as distance_matrix with top_k)

        Circuits are evaluated once; blending across λ values is cheap.

        Returns:
            dict {λ: [M, N] numpy float32 distance array}
        """
        from utils.metrics import euclidean_distance

        self.pre_net.cpu()
        qf_cpu = qf.cpu().float()
        gf_cpu = gf.cpu().float()

        with torch.no_grad():
            qf_enc = self.encode(qf_cpu)   # [M, n_q]
            gf_enc = self.encode(gf_cpu)   # [N, n_q]

        M, N = qf_enc.shape[0], gf_enc.shape[0]
        eucl = euclidean_distance(qf.cpu(), gf.cpu())   # [M, N]

        # Per-query Euclidean normalised to [0, 1].
        eucl_norm = np.empty_like(eucl)
        for i in range(M):
            lo, hi = eucl[i].min(), eucl[i].max()
            eucl_norm[i] = (eucl[i] - lo) / (hi - lo + 1e-8)

        # Compute circuit values for all top-K pairs once.
        k_store = np.zeros((M, top_k), dtype=np.float32)
        top_k_idx_store = np.empty((M, top_k), dtype=np.int64)
        for i in range(M):
            if i % 50 == 0:
                print(f"\r  Blended kernel top-{top_k}: query {i}/{M}", end="", flush=True)
            idx = np.argsort(eucl[i])[:top_k]
            top_k_idx_store[i] = idx
            for t, j in enumerate(idx):
                k_store[i, t] = self._circuit(qf_enc[i], gf_enc[int(j)])[0].item()
        print()

        # Build distance matrix for each λ (cheap — no circuit calls).
        results = {}
        for lam in lambdas:
            dist = eucl_norm + 1.0   # non-top-K baseline: [1, 2]
            for i in range(M):
                idx = top_k_idx_store[i]
                for t, j in enumerate(idx):
                    dist[i, int(j)] = (
                        (1.0 - lam) * eucl_norm[i, int(j)] + lam * (1.0 - k_store[i, t])
                    )
            results[lam] = dist.astype(np.float32)

        return results

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, n_qubits={self.n_qubits}"
