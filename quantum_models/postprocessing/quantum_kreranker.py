"""
QuantumKReciprocalReranker — VQC on k-NN distance patterns for re-ranking.

Unlike the pairwise reranker (which compares raw 768-dim features, causing
quantum concentration), this model operates on the structural distance
patterns of query and gallery relative to a shared k-NN neighbourhood:

    v_q[k]: distances from query q to its k-NN gallery neighbours
    v_g[k]: distances from gallery candidate g to the same k neighbours

    cat(v_q, v_g) [2k]
    → pre_net Linear(2k, n_qubits)    5:1 compression (vs 192:1 for raw feats)
    → sigmoid * π
    → AngleEmbedding + SEL VQC
    → probs [2^n_qubits]
    → out Linear(2^n_qubits, 1) + sigmoid
    → match_score [B] ∈ (0, 1)

If two tracklets are the same identity their distance patterns to the same
neighbourhood should look similar — the VQC learns this structural similarity.
"""

import torch
import torch.nn as nn
import pennylane as qml


class QuantumKReciprocalReranker(nn.Module):
    def __init__(self, k: int = 20, n_qubits: int = 8, n_layers: int = 2):
        super().__init__()
        self.k = k
        self.n_qubits = n_qubits

        self.pre_net = nn.Linear(2 * k, n_qubits, bias=True)
        nn.init.kaiming_normal_(self.pre_net.weight, nonlinearity='sigmoid')
        nn.init.zeros_(self.pre_net.bias)

        dev = qml.device('default.qubit', wires=n_qubits)
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.qlayer_weights = nn.Parameter(torch.randn(weight_shape) * 0.01)

        @qml.qnode(dev, diff_method='backprop', interface='torch')
        def circuit(angles, weights):
            qml.AngleEmbedding(angles, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

        self._circuit = circuit

        self.out = nn.Linear(2 ** n_qubits, 1, bias=True)
        nn.init.normal_(self.out.weight, std=0.001)
        nn.init.zeros_(self.out.bias)

        self._cpu_params = ['qlayer_weights']

    def _apply(self, fn):
        cpu_state = {k: getattr(self, k).data.clone() for k in self._cpu_params
                     if hasattr(self, k) and isinstance(getattr(self, k), nn.Parameter)}
        super()._apply(fn)
        for k, v in cpu_state.items():
            getattr(self, k).data = v
        return self

    def forward(self, v_q, v_g):
        """
        v_q: [B, k] distances from query to its k-NN neighbours
        v_g: [B, k] distances from gallery candidate to same k neighbours
        returns: [B] match scores in (0, 1)
        """
        x = torch.cat([v_q, v_g], dim=-1)                          # [B, 2k]
        angles = torch.sigmoid(self.pre_net(x)) * torch.pi         # [B, n_q]
        probs = self._circuit(angles.float().cpu(),
                              self.qlayer_weights.float())          # [B, 2^n_q]
        probs = probs.to(dtype=x.dtype, device=x.device)
        return torch.sigmoid(self.out(probs)).squeeze(-1)           # [B]
