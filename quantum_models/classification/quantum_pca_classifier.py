"""
quantum_models/classification/quantum_pca_classifier.py

Quantum PCA Classifier — two-stage quantum pipeline:

Stage 1 — QPCA (quantum dimensionality reduction):
    AmplitudeEmbedding(in_features → 2^n_qubits_pca amplitudes)
    StronglyEntanglingLayers  (learns principal component rotation)
    Partial measurement of n_qubits_out qubits → 2^n_qubits_out values
    These are the quantum principal components of the input feature.

Stage 2 — VQC Classifier:
    AngleEmbedding(n_components values → n_qubits_cls qubits)
    StronglyEntanglingLayers  (learns discriminative mapping)
    qml.probs → [2^n_qubits_cls] → Linear(2^n_qubits_cls, num_classes)

Replaces the classical Linear(in_features→num_classes) head AND the
Linear(in_features→n_qubits) squashing bottleneck in one unified quantum pipeline.

Defaults:
    n_qubits_pca = 10  → 2^10=1024 amplitudes (fits 768-dim ViT features; padded)
    n_qubits_out = 3   → 2^3=8 QPCA principal components extracted
    n_qubits_cls = 8   → 8-qubit AngleEmbedding classifier (matches 8 QPCA outputs)
    n_layers     = 2   → depth for both circuits
"""

import math
import torch
import torch.nn as nn
import pennylane as qml


class QuantumPCAClassifier(nn.Module):
    """
    Two-stage quantum classifier: QPCA dimensionality reduction → VQC classification.

    Stage 1 (QPCA):
        AmplitudeEmbedding(in_features, n_qubits_pca) — full fidelity, no squashing
        → StronglyEntanglingLayers (quantum PCA rotation)
        → partial probs over n_qubits_out qubits → [B, 2^n_qubits_out] principal components

    Stage 2 (VQC Classifier):
        AngleEmbedding(principal_components * π, n_qubits_cls)
        → StronglyEntanglingLayers
        → probs [B, 2^n_qubits_cls] → Linear → [B, num_classes]

    Args:
        in_features   : Input feature dim (e.g. 768 for ViT-B/16).
        num_classes   : Number of output classes.
        n_qubits_pca  : Qubits for QPCA stage. 2^n >= in_features. Default 10.
        n_qubits_out  : Qubits to measure from QPCA → principal components. Default 3 → 8 values.
        n_qubits_cls  : Qubits for classifier VQC. Must equal 2^n_qubits_out. Default 8.
        n_layers      : SEL depth for both stages. Default 2.
        bypass_quantum: If True, replace both stages with Linear+ReLU (ablation).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        n_qubits_pca: int = 10,
        n_qubits_out: int = 3,
        n_qubits_cls: int = 8,
        n_layers: int = 2,
        bypass_quantum: bool = False,
    ):
        super().__init__()
        self.in_features   = in_features
        self.num_classes   = num_classes
        self.n_qubits_pca  = n_qubits_pca
        self.n_qubits_out  = n_qubits_out
        self.n_qubits_cls  = n_qubits_cls
        self.n_layers      = n_layers
        self.n_components  = 2 ** n_qubits_out   # e.g. 8
        self.n_measurements = 2 ** n_qubits_cls  # e.g. 256
        self.bypass_quantum = bypass_quantum

        assert 2 ** n_qubits_pca >= in_features, (
            f"2^n_qubits_pca={2**n_qubits_pca} must be >= in_features={in_features}"
        )
        assert n_qubits_cls == self.n_components, (
            f"n_qubits_cls ({n_qubits_cls}) must equal 2^n_qubits_out ({self.n_components}) "
            f"so all QPCA outputs can be angle-encoded 1:1 into the classifier circuit."
        )

        if not bypass_quantum:
            # ── Stage 1: QPCA circuit ──────────────────────────────────────
            dev_pca = qml.device("default.qubit", wires=n_qubits_pca)

            @qml.qnode(dev_pca, interface="torch", diff_method="backprop")
            def _qpca_circuit(features, weights):
                # features: [B, in_features] — padded to 2^n_qubits_pca
                # weights:  [n_layers, n_qubits_pca, 3]
                qml.AmplitudeEmbedding(
                    features, wires=range(n_qubits_pca),
                    normalize=True, pad_with=0.
                )
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits_pca))
                # Partial measurement: only first n_qubits_out qubits
                return qml.probs(wires=range(n_qubits_out))

            self.qpca_circuit = _qpca_circuit
            pca_weight_shape  = qml.StronglyEntanglingLayers.shape(
                n_layers=n_layers, n_wires=n_qubits_pca
            )
            self.qpca_weights = nn.Parameter(torch.zeros(pca_weight_shape))
            nn.init.normal_(self.qpca_weights, mean=0, std=0.01)

            # ── Stage 2: VQC classifier circuit ───────────────────────────
            dev_cls = qml.device("default.qubit", wires=n_qubits_cls)

            @qml.qnode(dev_cls, interface="torch", diff_method="backprop")
            def _cls_circuit(angles, weights):
                # angles:  [B, n_components] — QPCA outputs scaled to [0, π]
                # weights: [n_layers, n_qubits_cls, 3]
                qml.AngleEmbedding(angles, wires=range(n_qubits_cls), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits_cls))
                return qml.probs(wires=range(n_qubits_cls))

            self.cls_circuit = _cls_circuit
            cls_weight_shape  = qml.StronglyEntanglingLayers.shape(
                n_layers=n_layers, n_wires=n_qubits_cls
            )
            self.cls_weights = nn.Parameter(torch.zeros(cls_weight_shape))
            nn.init.normal_(self.cls_weights, mean=0, std=0.01)

        else:
            # Classical ablation: two linear layers mirroring the two VQC stages
            self.bypass_net = nn.Sequential(
                nn.Linear(in_features, self.n_components),
                nn.ReLU(),
                nn.Linear(self.n_components, self.n_measurements),
                nn.ReLU(),
            )

        self.head = nn.Linear(self.n_measurements, num_classes, bias=False)
        nn.init.normal_(self.head.weight, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]
        Returns:
            [B, num_classes] logits
        """
        if self.bypass_quantum:
            return self.head(self.bypass_net(x.float()))

        # Stage 1: QPCA — full-fidelity amplitude encoding + partial measurement
        pca_out = self.qpca_circuit(x.float(), self.qpca_weights).float()  # [B, n_components]

        # Scale QPCA probs (∈[0,1], sum=1) to angle range [0, π] for AngleEmbedding
        angles = pca_out * math.pi   # [B, n_components]

        # Stage 2: VQC classifier on quantum principal components
        cls_out = self.cls_circuit(angles, self.cls_weights).float()  # [B, n_measurements]

        return self.head(cls_out)   # [B, num_classes]

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}→QPCA({self.n_qubits_pca}q,{self.n_layers}L)"
            f"→{self.n_components}→VQC({self.n_qubits_cls}q,{self.n_layers}L)"
            f"→{self.n_measurements}→{self.num_classes}"
        )
