"""
ClassicalPairwiseReranker — MLP ablation for QuantumPairwiseReranker.

Mirrors the quantum reranker structure exactly, replacing the VQC with a
classical MLP of matched depth and hidden dimensionality:

    cat(f_query, f_gallery) [B, 2*in_features]
    → pre_net Linear(2*D, n_qubits)    same compression as quantum
    → ReLU
    → hidden  Linear(n_qubits, 2^n_qubits)   same width as VQC output
    → ReLU
    → out     Linear(2^n_qubits, 1) + sigmoid
    → match_score [B] ∈ (0, 1)

Ablation purpose: isolate whether any reranking gain is from the quantum
circuit specifically, or just from the additional pairwise computation step.
"""

import torch
import torch.nn as nn


class ClassicalPairwiseReranker(nn.Module):
    def __init__(self, in_features=768, n_qubits=8, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        hidden_dim = 2 ** n_qubits  # matches VQC output dimensionality

        # Same compression as quantum pre_net
        self.pre_net = nn.Linear(in_features * 2, n_qubits, bias=True)
        nn.init.kaiming_normal_(self.pre_net.weight, nonlinearity='relu')
        nn.init.zeros_(self.pre_net.bias)

        # Classical expansion: mirrors quantum probs [2^n_qubits]
        self.hidden = nn.Linear(n_qubits, hidden_dim, bias=True)
        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')
        nn.init.zeros_(self.hidden.bias)

        # Output head: same as quantum
        self.out = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.normal_(self.out.weight, std=0.001)
        nn.init.zeros_(self.out.bias)

    def forward(self, fq, fg):
        """
        fq: [B, D] query features
        fg: [B, D] gallery candidate features
        returns: [B] match scores in (0, 1)
        """
        x = torch.cat([fq, fg], dim=-1)         # [B, 2D]
        h = torch.relu(self.pre_net(x))          # [B, n_qubits]
        h = torch.relu(self.hidden(h))           # [B, 2^n_qubits]
        score = torch.sigmoid(self.out(h)).squeeze(-1)  # [B]
        return score
