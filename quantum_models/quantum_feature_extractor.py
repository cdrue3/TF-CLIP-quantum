"""
quantum_models/quantum_feature_extractor.py

Quantum Feature Extractor — Parallel Integration Pattern.

Survey paper §3.2.2 (Feature Extraction level) + §3.3.2 (Parallel pattern):
"Parallel networks offer resilience against quantum noise and improved performance."
"Quantum feature extraction [...] enriches classical representations."

Architecture (QuantumAugmentedClassifier):
    x [B, in_features]
    ├──────────────────────────────────────────────────────► classical branch
    │                                                         (original features)
    └──► pre_net : Linear(in_features → n_qubits)             quantum branch
      → scaling  : sigmoid(x) * pi                            maps to (0, π)
      → qlayer   : AngleEmbedding + StronglyEntanglingLayers   VQC
      → probs()  : [B, 2^n_qubits]                            quantum features
                                                              ▼
    concat([x_original, quantum_feat])  →  [B, in_features + 2^n_qubits]
    → post_net : Linear(in_features + 2^n_qubits → num_classes)
    → [B, num_classes]

Why this is different from QuantumClassifier (sequential pattern):
    QuantumClassifier (sequential): classical compresses 768→n_qubits, ALL
    information flows through the quantum bottleneck. Classical overshadowing
    risk: post_net effectively undoes the quantum work.

    QuantumAugmentedClassifier (parallel): quantum provides ADDITIONAL features
    on top of the original CLIP embedding. The VQC is additive — it cannot
    hurt the classical path, only enrich it. This directly implements the
    "parallel" integration pattern recommended by the survey.

Key distinction:
    - x_original (768-dim CLIP features) is preserved and passed directly to post_net
    - quantum_feat (2^n_qubits-dim) is concatenated alongside
    - post_net sees: [classical context (768) || quantum enrichment (256)] = 1024-dim
    - The linear classifier can ignore quantum features if they are uninformative

This avoids the "classical overshadowing" problem because the VQC never controls
the entire information flow — it supplements rather than replaces.

Install:
    pip install pennylane==0.33.1
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumFeatureExtractor(nn.Module):
    """
    Extracts quantum features from classical input using a small VQC.

    Produces a [B, 2^n_qubits] feature vector that can be concatenated with
    classical features for downstream classification.  Does NOT produce logits —
    this is a feature extraction module, not a classifier.

    Args:
        in_features  (int): Input feature dimension (e.g. 768 or 512).
        n_qubits     (int): Number of qubits. Default 8 → 256 output features.
        n_layers     (int): StronglyEntanglingLayers depth. Default 2.
        device_name  (str): PennyLane device. Default 'default.qubit' (CPU sim).
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_quantum_features = 2 ** n_qubits

        # ------------------------------------------------------------------
        # Classical pre-projection: compresses in_features → n_qubits angles.
        # ------------------------------------------------------------------
        self.pre_net = nn.Linear(in_features, n_qubits, bias=False)

        # ------------------------------------------------------------------
        # Variational quantum circuit — same design as QuantumClassifier:
        #   AngleEmbedding: RY(sigmoid(x_i)·π) on each qubit
        #   StronglyEntanglingLayers: SU(2) rotations + long-range CNOT
        #   qml.probs(): full 2^n_qubits probability distribution
        # ------------------------------------------------------------------
        dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

        self._init_weights()

    def _init_weights(self):
        # pre_net: kaiming fan_in — output std ≈ sqrt(2/in_features) ≈ 0.051 for 768.
        # sigmoid(0.051)·π ≈ π/2 — near maximum gradient point.
        nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        # qlayer: near-identity init — std=0.01 keeps RY gates near-identity,
        # preventing barren plateau at init.
        nn.init.normal_(self.qlayer.weights, mean=0, std=0.01)

    def to(self, *args, **kwargs):
        """Pin qlayer to CPU; pre_net can move freely to GPU."""
        super().to(*args, **kwargs)
        self.qlayer.to(device=torch.device("cpu"), dtype=torch.float32)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]  (float32 or float16, any device)
        Returns:
            quantum_feat: [B, 2^n_qubits]  (same dtype and device as input)
        """
        input_dtype  = x.dtype
        input_device = x.device

        x = x.float()
        x = self.pre_net(x)                    # [B, n_qubits]  on input_device
        x = torch.sigmoid(x) * math.pi         # (0, π)

        # PennyLane runs on CPU.
        x = x.cpu().float()                    # [B, n_qubits]  on CPU, float32
        x = self.qlayer(x)                     # [B, 2^n_qubits]  on CPU
        x = x.to(input_device)                 # [B, 2^n_qubits]  on input_device

        return x.to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, n_quantum_features={self.n_quantum_features}"
        )


class QuantumAugmentedClassifier(nn.Module):
    """
    Parallel hybrid classifier: classical features + VQC-extracted quantum features.

    Wraps QuantumFeatureExtractor + nn.Linear for the fused representation.
    Drop-in replacement for nn.Linear(in_features, num_classes) with quantum enrichment.

    Named with 'classifier' prefix so the LR boosting logic in train_qfeatext.py
    correctly identifies it by parameter name.

    Args:
        in_features    (int):  Input feature dimension (e.g. 768 or 512).
        num_classes    (int):  Number of output identity classes.
        n_qubits       (int):  Number of qubits for the VQC. Default 8.
        n_layers       (int):  StronglyEntanglingLayers depth. Default 2.
        device_name    (str):  PennyLane device. Default 'default.qubit'.
        bypass_quantum (bool): If True, replace the entire parallel VQC path with a
                               pure nn.Linear(in_features, num_classes) — a proper
                               linear probe ablation with zero bottleneck and no quantum.
                               Used to isolate whether qfeatext gains come from the
                               parallel architecture or from the quantum features.
                               Default False (use quantum circuit).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        device_name: str = "default.qubit",
        bypass_quantum: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.n_quantum_features = 2 ** n_qubits
        self.bypass_quantum = bypass_quantum

        if bypass_quantum:
            # Linear probe: pure nn.Linear(in_features→num_classes), no VQC, no concat.
            # This is the correct baseline for qfeatext — no bottleneck, no quantum.
            # Named 'post_net' so the LR boost logic in train_qfeatext.py still applies.
            self.fused_dim = in_features
            self.post_net = nn.Linear(in_features, num_classes, bias=False)
        else:
            self.fused_dim = in_features + self.n_quantum_features

            # VQC for quantum feature extraction
            self.quantum_extractor = QuantumFeatureExtractor(
                in_features, n_qubits=n_qubits, n_layers=n_layers, device_name=device_name,
            )

            # Classical classifier on the fused [classical || quantum] representation.
            # Named 'post_net' to match the LR boost naming convention in train_qfeatext.py.
            self.post_net = nn.Linear(self.fused_dim, num_classes, bias=False)

        self._init_post_net()

    def _init_post_net(self):
        # kaiming_uniform with fan_in = fused_dim (or in_features for linear probe).
        nn.init.kaiming_uniform_(self.post_net.weight, a=math.sqrt(5))

    def to(self, *args, **kwargs):
        """Delegate to() to quantum_extractor which handles CPU-pinning."""
        super().to(*args, **kwargs)
        if not self.bypass_quantum:
            self.quantum_extractor.to(*args, **kwargs)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]  (float32 or float16, any device – AMP safe)
        Returns:
            logits: [B, num_classes]  (same dtype and device as input)
        """
        input_dtype  = x.dtype

        if self.bypass_quantum:
            # Pure linear probe: no VQC, no concat, no bottleneck.
            return self.post_net(x.float()).to(input_dtype)

        x_classical = x.float()                          # preserve original features
        quantum_feat = self.quantum_extractor(x_classical)  # [B, 2^n_qubits]

        # Concatenate classical + quantum features.
        fused = torch.cat([x_classical, quantum_feat], dim=1)   # [B, fused_dim]

        logits = self.post_net(fused)                    # [B, num_classes]
        return logits.to(input_dtype)

    def extra_repr(self) -> str:
        if self.bypass_quantum:
            return (
                f"in_features={self.in_features}, num_classes={self.num_classes}, "
                f"mode=linear_probe (bypass_quantum=True)"
            )
        return (
            f"in_features={self.in_features}, num_classes={self.num_classes}, "
            f"n_qubits={self.n_qubits}, fused_dim={self.fused_dim}"
        )
