"""
loss/quantum_triplet_loss.py

Quantum kernel triplet loss — Optimisation stage quantum component.

Replaces Euclidean distance in triplet loss with a trainable quantum kernel:

    K(x_i, x_j) = <φ(x_i) | φ(x_j)>   (cosine over probability vectors)
    d(x_i, x_j) = 1 − K(x_i, x_j)     (kernel-induced distance)

    φ(x): x [B,768] → pre_net [768→n_qubits] → AngleEmbedding
              → StronglyEntanglingLayers (trainable) → probs [2^n_qubits]

Training-only: the loss shapes the feature space geometry during training.
At inference, standard L2 / cosine distance is used for gallery retrieval.

---- Integration ----------------------------------------------------------
QuantumTripletLoss is an nn.Module with two sets of trainable parameters:
  • pre_net.weight   — 768→n_qubits linear projection
  • q_weights        — VQC rotation angles

These are outside the main model, so they need their own optimizer update.

Two ways to wire this up:

  Option A — attach to model (simplest, no processor changes):
      model.q_triplet = QuantumTripletLoss(feat_dim=768)
      # model.parameters() now includes q_triplet params
      # Same optimizer, same scheduler — everything automatic

  Option B — separate optimizer (more control):
      q_triplet = QuantumTripletLoss(feat_dim=768)
      optimizer_qk = torch.optim.Adam(q_triplet.parameters(), lr=1e-3)
      # After loss.backward():
      #   optimizer_qk.step(); optimizer_qk.zero_grad()

Use make_loss.py's `make_loss_q_triplet` helper to create the loss_func that
calls q_triplet internally:
    loss_func, center_criterion, q_triplet = make_loss_q_triplet(cfg, num_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

from .triplet_loss import hard_example_mining


class QuantumTripletLoss(nn.Module):
    """
    Drop-in replacement for TripletLoss using a trainable quantum kernel distance.

    Args:
        feat_dim   (int): Input feature dimension. Default 768 (TF-CLIP ViT-B/16).
        n_qubits   (int): Qubits in VQC. Default 6 → 2^6=64 dim probability vector.
                          Keep ≤ 8; barren plateau risk grows sharply above this.
        n_layers   (int): StronglyEntanglingLayers. Default 1 (safe for gradients).
        margin  (float): Triplet margin. None → soft margin loss (recommended).
        device_name(str): PennyLane backend.
    """

    def __init__(self, feat_dim: int = None, n_qubits: int = 6,
                 n_layers: int = 1, margin: float = None,
                 device_name: str = 'default.qubit'):
        super().__init__()
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.margin    = margin
        self._feat_dim = feat_dim  # None = auto-detect on first forward

        # pre_net built lazily on first call if feat_dim unknown
        if feat_dim is not None:
            self.pre_net = nn.Linear(feat_dim, n_qubits, bias=False)
            nn.init.normal_(self.pre_net.weight, std=1.0 / math.sqrt(feat_dim))
        else:
            self.pre_net = None

        # VQC trainable rotation angles — near-identity init avoids early barren plateau
        wshape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        self.q_weights = nn.Parameter(torch.zeros(wshape))
        nn.init.normal_(self.q_weights, std=0.01)

        dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def _circuit(angles, weights):
            # angles:  [B, n_qubits] — PennyLane broadcasts over batch dimension
            # weights: [n_layers, n_qubits, 3] — shared across batch
            qml.AngleEmbedding(angles, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))   # [B, 2^n_qubits]

        self._circuit = _circuit

        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

        n_params = feat_dim * n_qubits + int(torch.zeros(wshape).numel())
        print(f"[QuantumTripletLoss] n_qubits={n_qubits}, n_layers={n_layers}, "
              f"margin={margin}, trainable_params={n_params}")

    # ------------------------------------------------------------------
    def _kernel_dist(self, feats: torch.Tensor) -> torch.Tensor:
        """[B, D] features → [B, B] quantum kernel distance matrix."""
        # Lazy init pre_net on first call when feat_dim wasn't known at construction
        if self.pre_net is None:
            D = feats.shape[1]
            self.pre_net = nn.Linear(D, self.n_qubits, bias=False).to(feats.device)
            nn.init.normal_(self.pre_net.weight, std=1.0 / math.sqrt(D))
            print(f"[QuantumTripletLoss] auto-detected feat_dim={D}, built pre_net")
        angles = self.pre_net(feats.float())                     # [B, n_qubits]

        # Circuit runs on CPU (PennyLane default.qubit)
        probs  = self._circuit(angles.cpu(),
                               self.q_weights.cpu())             # [B, 2^n_q]
        probs  = probs.to(feats.device)

        # Normalise → cosine kernel: K_ij = cos(probs_i, probs_j), K_ii = 1
        q_norm = F.normalize(probs.float(), p=2, dim=1)          # [B, 2^n_q]
        K      = q_norm @ q_norm.T                               # [B, B] ∈ [−1, 1]

        return (1.0 - K).clamp(min=0.0)                          # [B, B] distances

    # ------------------------------------------------------------------
    def forward(self, feats: torch.Tensor,
                labels: torch.Tensor) -> tuple:
        """
        Args:
            feats:  [B, D] feature vectors (from model backbone)
            labels: [B]    identity labels

        Returns:
            (loss, dist_ap, dist_an) — identical interface to TripletLoss.__call__
        """
        dist_mat = self._kernel_dist(feats)                      # [B, B]
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss, dist_ap, dist_an
