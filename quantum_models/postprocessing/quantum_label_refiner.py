"""
quantum_models/postprocessing/quantum_label_refiner.py

Quantum Probabilistic Label Refining (QPLR).

After the classifier produces logits [B, num_classes], a VQC processes the
top-K class predictions and produces refined soft labels capturing quantum
inter-class correlations. This is a post-processing stage component applied
after inference but during training to refine the label distribution used
in the cross-entropy loss.

Particularly relevant for AG-VPReID: aerial/ground pairs create inherent
class ambiguity (same identity looks different across cameras), and the
standard cross-entropy treats all non-GT classes identically. QPLR learns
quantum correlations between visually similar classes.

Architecture:
    logits [B, num_classes]
    → top-K selector (default K=32) → top_logits [B, K]
    → pre_net: Linear(K → n_qubits=8)
    → sigmoid(·) * π → VQC → probs [B, 2^8=256]
    → refine_net: Linear(256 → K) + softmax → refined soft weights [B, K]
    → scatter back to full [B, num_classes] distribution
    → KL divergence(refined_dist, one_hot_target)

Total loss: standard_CE + kl_weight * KL_loss

bypass_quantum=True: KL loss is zero (standard CE only).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuantumLabelRefiner(nn.Module):
    """
    VQC-based label refiner that produces soft labels from classifier logits.

    Args:
        num_classes   (int): Total number of identity classes.
        top_k         (int): Number of top-K class logits to process via VQC. Default 32.
        n_qubits      (int): Qubit count. Default 8.
        n_layers      (int): VQC depth. Default 2.
        kl_weight     (float): Weight for the KL divergence loss. Default 0.5.
        bypass_quantum(bool): If True, return 0 KL loss (disable QPLR).
        device_name   (str): PennyLane device.
    """

    def __init__(
        self,
        num_classes: int,
        top_k: int = 32,
        n_qubits: int = 8,
        n_layers: int = 2,
        kl_weight: float = 0.5,
        bypass_quantum: bool = False,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.num_classes    = num_classes
        self.top_k          = min(top_k, num_classes)
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.kl_weight      = kl_weight
        self.n_measurements = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        # Compress top-K logits to quantum angle space
        self.pre_net = nn.Linear(self.top_k, n_qubits, bias=False)

        if not bypass_quantum:
            n_q = n_qubits
            dev = qml.device(device_name, wires=n_q)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _circuit(angles, weights):
                qml.AngleEmbedding(angles, wires=range(n_q), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                return qml.probs(wires=range(n_q))

            self.circuit = _circuit
            weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_q)
            self.qlayer_weights = nn.Parameter(torch.zeros(weight_shape))

        # Refine net: maps VQC probs → refined soft weights for top-K classes
        self.refine_net = nn.Linear(self.n_measurements, self.top_k)
        self._init_weights()

        print(
            f"[QPLR] num_classes={num_classes}, top_k={self.top_k}, "
            f"n_qubits={n_qubits}, n_layers={n_layers}, kl_weight={kl_weight}"
        )

    def _init_weights(self):
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        if not self.bypass_quantum:
            nn.init.normal_(self.qlayer_weights, mean=0, std=0.01)
        # Near-identity init: refined logits ≈ input logits at start
        nn.init.normal_(self.refine_net.weight, mean=0, std=0.01)
        nn.init.zeros_(self.refine_net.bias) if hasattr(self.refine_net, 'bias') and self.refine_net.bias is not None else None

    def _apply(self, fn):
        super()._apply(fn)
        if not self.bypass_quantum:
            self.qlayer_weights.data = self.qlayer_weights.data.cpu().float()
        return self

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute QPLR KL loss component.

        Args:
            logits  : [B, num_classes] — raw classifier output logits
            targets : [B] — integer identity labels (0-indexed)

        Returns:
            kl_loss: scalar — KL(refined_distribution || one_hot_target) * kl_weight
                     Returns 0 if bypass_quantum=True.
        """
        if self.bypass_quantum:
            return logits.new_zeros(1).squeeze()

        input_device = logits.device
        B = logits.shape[0]

        # Select top-K class logits (non-differentiable selector — straight-through)
        with torch.no_grad():
            topk_vals, topk_indices = logits.float().topk(self.top_k, dim=1)  # [B, K]

        # Use the logit values at top-K positions (detach indices, not values)
        top_logits = logits.gather(1, topk_indices.detach())  # [B, K] — differentiable

        # VQC processing
        angles = torch.sigmoid(self.pre_net(top_logits.float())) * math.pi  # [B, n_q]
        angles_cpu  = angles.cpu().float()
        weights_cpu = self.qlayer_weights.cpu().float()

        probs = self.circuit(angles_cpu, weights_cpu).float().to(input_device)  # [B, 2^n_q]

        # Refined soft weights for top-K classes
        refined_logits = self.refine_net(probs)  # [B, K]
        refined_weights = F.softmax(refined_logits, dim=1)  # [B, K] — soft label distribution

        # Build full [B, num_classes] soft distribution from top-K weights
        soft_dist = logits.new_zeros(B, self.num_classes)
        soft_dist.scatter_(1, topk_indices.detach(), refined_weights.to(soft_dist.dtype))  # [B, num_classes]

        # KL(one_hot || soft_dist) = -log(soft_dist[gt]) for a one-hot target (= NLL).
        # Avoids 0*log(0)=nan and float16 underflow that plagued F.kl_div.
        log_probs = soft_dist.float().clamp(min=1e-8).log()  # float32 for stability
        kl = F.nll_loss(log_probs, targets)

        return self.kl_weight * kl

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, top_k={self.top_k}, "
            f"n_qubits={self.n_qubits}, kl_weight={self.kl_weight}, "
            f"bypass={self.bypass_quantum}"
        )
