"""
QuantumPairwiseReranker — learned VQC similarity for post-hoc re-ranking.

Trained on frozen features from a classical checkpoint:
    cat(f_query, f_gallery) [B, 2*in_features]
    → pre_net Linear(2*D, n_qubits)    learned compression (NOT PCA)
    → sigmoid * π                       angle encoding
    → AngleEmbedding + SEL              VQC
    → probs [B, 2^n_qubits]
    → out Linear(2^n_qubits, 1) + sigmoid
    → match_score [B] ∈ (0, 1)

At inference: score each (query, top-K candidate) pair, rerank by score.
"""

import torch
import torch.nn as nn
import pennylane as qml


class QuantumPairwiseReranker(nn.Module):
    def __init__(self, in_features=768, n_qubits=8, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Learned compression — gradients flow end-to-end through this
        self.pre_net = nn.Linear(in_features * 2, n_qubits, bias=True)
        nn.init.kaiming_normal_(self.pre_net.weight, nonlinearity='sigmoid')
        nn.init.zeros_(self.pre_net.bias)

        # VQC
        dev = qml.device('default.qubit', wires=n_qubits)
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.qlayer_weights = nn.Parameter(
            torch.randn(weight_shape) * 0.01
        )

        @qml.qnode(dev, diff_method='backprop', interface='torch')
        def circuit(angles, weights):
            qml.AngleEmbedding(angles, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

        self._circuit = circuit

        # Output head: probs → match score
        self.out = nn.Linear(2 ** n_qubits, 1, bias=True)
        nn.init.normal_(self.out.weight, std=0.001)
        nn.init.zeros_(self.out.bias)

        # Pin VQC weights to CPU (PennyLane default.qubit is CPU-only)
        self._cpu_params = ['qlayer_weights']

    def _apply(self, fn):
        # Keep qlayer_weights on CPU regardless of .to(device) calls
        cpu_state = {k: getattr(self, k).data.clone() for k in self._cpu_params
                     if hasattr(self, k) and isinstance(getattr(self, k), nn.Parameter)}
        super()._apply(fn)
        for k, v in cpu_state.items():
            getattr(self, k).data = v
        return self

    def forward(self, fq, fg):
        """
        fq: [B, D] query features
        fg: [B, D] gallery candidate features
        returns: [B] match scores in (0, 1)
        """
        x = torch.cat([fq, fg], dim=-1)           # [B, 2D]
        angles = torch.sigmoid(self.pre_net(x)) * torch.pi   # [B, n_q]

        # Run VQC (on CPU)
        angles_cpu = angles.float().cpu()
        weights_cpu = self.qlayer_weights.float()
        probs = self._circuit(angles_cpu, weights_cpu)        # [B, 2^n_q]

        # Cast back to original dtype/device
        probs = probs.to(dtype=x.dtype, device=x.device)

        score = torch.sigmoid(self.out(probs)).squeeze(-1)    # [B]
        return score
