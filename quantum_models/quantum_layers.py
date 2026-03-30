"""
quantum_models/quantum_layers.py

Reusable quantum circuit layers for TF-CLIP integration.

QuantumClassifier
-----------------
A hybrid quantum-classical drop-in replacement for nn.Linear classifier heads.

Architecture per sample:
    x [B, in_features]
    -> pre_net  : Linear(in_features → n_qubits)               classical compression
    -> scaling  : sigmoid(x) * pi                               maps to angle range (0, π)
    -> qlayer   : AngleEmbedding + StronglyEntanglingLayers     variational quantum circuit
    -> post_net : Linear(2^n_qubits → num_classes)              classical output projection
    -> [B, num_classes]

Quantum circuit per sample:
    - AngleEmbedding (rotation='Y'): applies RY(x_i) on wire i  (single embedding)
    - StronglyEntanglingLayers:      n_layers of (SU(2) Rot gates + long-range CNOT)
                                     weight shape (n_layers, n_qubits, 3)
    - Measurement: qml.probs() over all 2^n_qubits basis states

Why StronglyEntanglingLayers over BasicEntanglerLayers:
    Ablation (bypass_quantum=True, 8q, 15 epochs) showed a classical Linear+ReLU
    with identical dimensions reached acc_id1=0.027 vs quantum BasicEntangler 0.015.
    BasicEntanglerLayers with near-identity init produces a near-linear circuit —
    structurally equivalent to what a classical Linear layer provides.
    StronglyEntanglingLayers uses SU(2) Rot(φ,θ,ω)=RZ(ω)·RY(θ)·RZ(φ) on each qubit
    and CNOT with range [1, 2, ..., n_layers] per layer, creating multi-qubit
    correlations not decomposable into a product state.

Full probability distribution measurement (qml.probs):
    Returns the probability of each computational basis state |b_{n-1}...b_0⟩.
    For n_qubits=8: 256 features; for n_qubits=10: 1024 features > 625 classes.
    In state-vector simulation (default.qubit, diff_method="backprop"), all 2^n
    amplitudes are already computed in memory — returning probs() costs nothing
    extra per circuit evaluation compared to single-qubit expectations.
    The full probability vector captures all entanglement structure that single-qubit
    Pauli expectations discard, giving post_net a much richer feature space to
    carve 625 linearly-separated regions from.

Why sigmoid (not tanh) for angle scaling:
    tanh(x)·π has a null gradient at x=0 (sin(tanh(0)·π) = sin(0) = 0).
    sigmoid(0) = 0.5 maps to θ = π/2, where d⟨PauliZ⟩/dθ = −sin(π/2) = −1 (maximum).
    Critically, pre_net output concentrates near 0 in normal operation, so sigmoid
    places the mean embedding at the maximum-gradient point of the PauliZ observable.

    IMPORTANT: data re-uploading (re-embedding at each variational layer) with std=0.01
    (near-identity VQC) REINTRODUCES the null gradient.  With n_layers=2 and std=0.01,
    the VQC between two embeddings barely rotates the state, so the effective embedding
    sum ≈ 2θ ≈ π at x=0 → d⟨Z⟩/dx = −2sin(π) = 0 — same null as tanh.
    Fix (survey §3.3.3): use std=0.2 for VQC weights — non-trivial rotations between
    embeddings break the null-gradient alignment.  See "reuploading" encoding below.
    For single embedding (angle, dense_angle, iqp): std=0.01 remains correct.

Design notes:
    - diff_method="backprop" uses PyTorch autograd through the state-vector
      simulation; compatible with default.qubit on CPU.
    - Inputs are cast to float32 before the circuit (AMP/fp16 safety) and
      cast back to the caller's dtype on return.
    - PennyLane 0.33.1 required (latest version compatible with Python 3.8).
      Tested with default.qubit (CPU simulator).
      For faster CPU simulation: pip install pennylane-lightning==0.33.1
      and pass device_name="lightning.qubit".

Install:
    pip install pennylane==0.33.1
"""

import math

import torch
import torch.nn as nn
import pennylane as qml


class QuantumClassifier(nn.Module):
    """
    Hybrid quantum-classical classifier.

    Replaces an nn.Linear(in_features, num_classes) head with a three-stage
    pipeline:  classical pre-projection -> VQC -> classical post-projection.

    Args:
        in_features    (int):  Input feature dimension (e.g. 768 or 512).
        num_classes    (int):  Number of output identity classes.
        n_qubits       (int):  Number of qubits.  Default 8.
                               Determines the bottleneck dimension of the quantum layer.
        n_layers       (int):  Variational entangler layer depth.  Default 2.
        device_name    (str):  PennyLane device string.  Default "default.qubit" (CPU).
                               Use "lightning.qubit" for faster CPU or
                               "lightning.gpu"    for GPU (separate install required).
        bypass_quantum (bool): If True, replace the VQC with a classical
                               Linear(n_qubits→n_measurements)+ReLU layer.
                               Used for ablation: tests whether the quantum circuit
                               adds anything beyond the classical pre/post projections.
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
        encoding: str = "angle",
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_measurements = 2 ** n_qubits   # full probability distribution over basis states
        self.bypass_quantum = bypass_quantum
        self.encoding = encoding
        # ------------------------------------------------------------------
        # 1. Classical pre-projection
        #    Compresses in_features -> n_qubits so each value can be used as
        #    a rotation angle in the quantum circuit (or classical equivalent).
        #    reuploading: n_layers INDEPENDENT pre_nets, one per VQC layer.
        #    Each learns a different linear projection of the CLIP features;
        #    the same CLIP vector is fed to all (data re-uploading §3.2.1).
        # ------------------------------------------------------------------
        if encoding == "reuploading":
            self.pre_net = None  # replaced by self.pre_nets (ModuleList)
            self.pre_nets = nn.ModuleList([
                nn.Linear(in_features, n_qubits, bias=False) for _ in range(n_layers)
            ])
            _pre_net_out = n_qubits  # each pre_net output; qlayer input is n_layers * n_qubits
        else:
            # pre_net output dim: dense_angle encodes 2 features per qubit (angle + phase);
            # angle and iqp encode 1 feature per qubit.
            _pre_net_out = 2 * n_qubits if encoding == "dense_angle" else n_qubits
            self.pre_net = nn.Linear(in_features, _pre_net_out, bias=False)

        if bypass_quantum:
            # ------------------------------------------------------------------
            # 2a. Classical ablation path (bypass_quantum=True)
            #     Replaces the VQC with Linear(n_qubits→n_measurements) + ReLU.
            #     Same input/output dimensions as the quantum path; no PennyLane.
            #     Used to test whether the VQC adds anything beyond the
            #     classical pre/post projections with the same bottleneck width.
            # ------------------------------------------------------------------
            self.classical_expansion = nn.Sequential(
                nn.Linear(_pre_net_out, self.n_measurements, bias=False),
                nn.ReLU(),
            )
        else:
            # ------------------------------------------------------------------
            # 2b. Variational quantum circuit (VQC) — single embedding, full probs measurement
            #     AngleEmbedding fires once, then n_layers of StronglyEntanglingLayers.
            #     Single embedding is required for the sigmoid gradient fix to hold
            #     (see module docstring for the full derivation).
            #
            #     Why StronglyEntanglingLayers over BasicEntanglerLayers:
            #     BasicEntanglerLayers = RX + nearest-neighbour CNOT ring.
            #     With near-identity init (std=0.01), this is approximately linear,
            #     producing probs ≈ products of cos²/sin² of individual angles —
            #     structurally similar to what a classical Linear+ReLU produces.
            #     Ablation (bypass_quantum=True, 8q, 15 epochs) showed classical
            #     Linear+ReLU reached acc_id1=0.027 vs quantum BasicEntangler 0.015:
            #     the VQC was not competitive.
            #
            #     StronglyEntanglingLayers = SU(2) rotation (Rot gate: RZ·RY·RZ) on
            #     each qubit + CNOT with range [1, 2, ..., n_layers] per layer.
            #     Range>1 entanglement creates correlations across non-adjacent qubits
            #     that are not decomposable into a product state — genuinely hard to
            #     replicate with a classical Linear layer of the same width.
            #     Weight shape: (n_layers, n_qubits, 3)  (3 angles per qubit per layer).
            #
            #     Measurement: qml.probs() → 2^n_qubits probabilities over all basis states.
            #     All amplitudes are already in memory for state-vector simulation, so
            #     returning probs() costs nothing extra vs single-qubit expectations.
            # ------------------------------------------------------------------
            dev = qml.device(device_name, wires=n_qubits)

            if encoding == "dense_angle":
                # Dense Angle Encoding (survey paper Table 3, §3.2.1):
                #   |ψ_j⟩ = cos(π·x_{2j-1})|0⟩ + e^{i·2π·x_{2j}} sin(π·x_{2j-1})|1⟩
                # Encodes 2 features per qubit via RY (rotation) + PhaseShift (phase).
                # inputs: [..., 2*n_qubits] — angles in [..., :n_qubits], phases in [..., n_qubits:]
                # Use inputs[..., q] (ellipsis) instead of inputs[q] so that PennyLane's
                # broadcast mode correctly indexes FEATURES (columns), not batch samples (rows).
                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def _circuit(inputs, weights):
                    qml.AngleEmbedding(inputs[..., :n_qubits], wires=range(n_qubits), rotation="Y")
                    for q in range(n_qubits):
                        qml.PhaseShift(inputs[..., n_qubits + q], wires=q)
                    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                    return qml.probs(wires=range(n_qubits))

            elif encoding == "iqp":
                # IQP Embedding (Instantaneous Quantum Polynomial, survey Table 3):
                #   H(x) = Σ x_i Z_i + Σ x_i·x_j Z_i⊗Z_j — second-order feature interactions.
                # PennyLane: IQPEmbedding applies RZ(x_i) and RZZ(x_i·x_j) for all pairs.
                # inputs: [n_qubits] — scaled features in (0, π).
                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def _circuit(inputs, weights):
                    qml.IQPEmbedding(inputs, wires=range(n_qubits), n_repeats=1)
                    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                    return qml.probs(wires=range(n_qubits))

            elif encoding == "reuploading":
                # Data re-uploading (survey §3.2.1 + §3.3.3):
                # Independent pre_nets supply n_qubits angles each; they are concatenated
                # as inputs[..., l*n_qubits:(l+1)*n_qubits] for layer l.
                # Interleaved embed → entangle → embed → entangle ... ensures the VQC state
                # is non-trivial at each re-embedding step, which is required for data
                # re-uploading to increase expressivity beyond a single embedding.
                # inputs: [B, n_layers * n_qubits]
                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def _circuit(inputs, weights):
                    for l in range(n_layers):
                        qml.AngleEmbedding(
                            inputs[..., l * n_qubits : (l + 1) * n_qubits],
                            wires=range(n_qubits), rotation="Y",
                        )
                        qml.StronglyEntanglingLayers(weights[l : l + 1], wires=range(n_qubits))
                    return qml.probs(wires=range(n_qubits))

            else:  # 'angle' (default) — single AngleEmbedding, backward compatible
                # inputs shape: [n_qubits]  (TorchLayer handles the batch loop)
                # Single angle embedding: RY(x_i) on each qubit i.
                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def _circuit(inputs, weights):
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                    return qml.probs(wires=range(n_qubits))

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            self.qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

        # ------------------------------------------------------------------
        # 3. Classical post-projection
        #    Maps 2^n_qubits features -> num_classes logits.
        # ------------------------------------------------------------------
        self.post_net = nn.Linear(self.n_measurements, num_classes, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        # pre_net: use fan_in so output std ≈ sqrt(2/in_features).
        # fan_out gives std = sqrt(2/n_qubits) = sqrt(2/8) ≈ 0.5, which
        # produces pre_net output std ≈ sqrt(768) × 0.5 ≈ 14 → sigmoid
        # saturates for large inputs.  fan_in gives std ≈ sqrt(2/768) ≈ 0.051
        # → output std ≈ 1.4 → sigmoid operates in its linear regime, and
        # the mean embedding sits near sigmoid(0)·π = π/2 (equator), where
        # the PauliZ gradient is maximised.
        if self.encoding == "reuploading":
            for pn in self.pre_nets:
                nn.init.kaiming_normal_(pn.weight, a=0, mode="fan_in")
        else:
            nn.init.kaiming_normal_(self.pre_net.weight, a=0, mode="fan_in")
        # post_net: kaiming_uniform (PyTorch default for nn.Linear).
        # With fan_in = 2^n_qubits (e.g. 256 for n_qubits=8), kaiming gives
        # std ≈ 1/√256 = 0.0625 and logit std = O(1) since
        # logit_var = fan_in × std² = 256 × (1/256) = 1.
        nn.init.kaiming_uniform_(self.post_net.weight, a=math.sqrt(5))
        if self.bypass_quantum:
            # classical_expansion: kaiming_normal_(fan_in) for the Linear layer.
            nn.init.kaiming_normal_(self.classical_expansion[0].weight, a=0, mode="fan_in")
        else:
            if self.encoding == "reuploading":
                # std=0.2: non-trivial rotations between embedding layers → avoids null gradient.
                # Survey §3.3.3: with std=0.01 (near-identity), the VQC between two embeddings
                # barely rotates the state, so the second embedding adds angles to the same
                # axis as the first → effective angle ≈ 2θ ≈ π at sigmoid init → gradient=0.
                # std=0.2 creates ~0.2 rad rotations, mixing qubits between embeddings so the
                # state is not aligned with the null-gradient axis at the second embedding.
                nn.init.normal_(self.qlayer.weights, mean=0, std=0.2)
            else:
                # qlayer: near-identity initialization (std=0.01).
                # PennyLane TorchLayer default init gives std≈3.23, placing most weights
                # near w≈π where sin(w)≈0 → near-zero gradients ("barren plateau").
                # std=0.01 keeps RX gates near-identity, preserving the structure set
                # by AngleEmbedding and giving O(1) output variance from step 1.
                nn.init.normal_(self.qlayer.weights, mean=0, std=0.01)

    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):
        """Override to() to keep the quantum layer pinned to CPU.

        PennyLane's default.qubit simulator runs state-vector operations on
        CPU.  When the model is moved to CUDA via model.to("cuda"), all
        nn.Parameters — including the TorchLayer VQC weights — would normally
        migrate to GPU, causing a device mismatch at circuit execution time.

        This override calls the parent to() for pre_net and post_net (so they
        can live on GPU), then immediately forces qlayer back to CPU float32.
        In bypass_quantum mode the classical_expansion lives on the same device
        as pre_net/post_net (GPU), so no pinning is needed.
        """
        # Move everything via the normal nn.Module logic.
        super().to(*args, **kwargs)
        if not self.bypass_quantum:
            # Pin qlayer back to CPU float32.
            # float32 is required by default.qubit; CPU is required by the simulator.
            self.qlayer.to(device=torch.device("cpu"), dtype=torch.float32)
        return self

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]  (float32 or float16, any device – AMP safe)
        Returns:
            logits: [B, num_classes]  (same dtype and device as input)
        """
        input_dtype   = x.dtype
        input_device  = x.device

        # AMP guard: cast to float32 for numerical stability.
        x = x.float()

        if self.encoding == "reuploading":
            # Data re-uploading (survey §3.2.1 + §3.3.3):
            # Each pre_net sees the same CLIP features and learns a different projection.
            # sigmoid·π maps each output to (0, π) angles; outputs are concatenated.
            # The VQC interleaves embed + entangle per layer (see __init__ for circuit).
            chunks = [torch.sigmoid(pn(x)) * math.pi for pn in self.pre_nets]
            x = torch.cat(chunks, dim=1).cpu().float()   # [B, n_layers * n_q] on CPU
            x = self.qlayer(x).to(input_device)          # [B, 2^n_q] on input_device
        else:
            # Stage 1 – classical compression (runs on input device, e.g. CUDA)
            # dense_angle: pre_net outputs 2*n_qubits; angle/iqp: n_qubits.
            x = self.pre_net(x)                              # [B, n_qubits or 2*n_qubits]

            # Stage 2 – encoding-specific scaling + quantum circuit (or classical ablation).
            if self.encoding == "dense_angle":
                # Split into rotation angles (0, π) and phases (0, 2π).
                # Angles: sigmoid(0)·π = π/2 — maximum d⟨Z⟩/dθ (same as single encoding).
                # Phases: sigmoid(0)·2π = π — non-zero PhaseShift gradient everywhere.
                angles = torch.sigmoid(x[:, :self.n_qubits]) * math.pi       # [B, n_qubits]
                phases = torch.sigmoid(x[:, self.n_qubits:]) * 2 * math.pi   # [B, n_qubits]
                x = torch.cat([angles, phases], dim=1)                        # [B, 2*n_qubits]
            else:
                # 'angle' and 'iqp': sigmoid·π scaling — places mean at π/2 (max gradient).
                x = torch.sigmoid(x) * math.pi                                # [B, n_qubits]

            if self.bypass_quantum:
                # Stage 2a – classical ablation: Linear(pre_net_out → n_measurements) + ReLU
                # Runs on input_device (GPU); no device bridge needed.
                x = self.classical_expansion(x)              # [B, n_measurements]  on input_device
            else:
                # Stage 2b – quantum circuit
                # Device bridge: PennyLane default.qubit runs on CPU.
                # .cpu() and .to(device) are differentiable — gradients flow correctly.
                # NOTE: re-cast to float32 here — autocast causes pre_net to return float16
                # even after the explicit float() above; float16 → ComplexHalf state vectors
                # in PennyLane → imaginary-part gradients silently discarded → VQC frozen.
                x = x.cpu().float()                          # [B, n_qubits or 2*n_qubits] on CPU
                x = self.qlayer(x)                           # [B, 2^n_qubits]  on CPU
                x = x.to(input_device)                       # [B, 2^n_qubits]  on input_device

        # Stage 3 – classical output projection (runs on input device)
        x = self.post_net(x)                   # [B, num_classes]

        # Restore original dtype so downstream loss functions see consistent types.
        return x.to(input_dtype)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        mode = "classical_ablation" if self.bypass_quantum else f"n_layers={self.n_layers}"
        return (
            f"in_features={self.in_features}, num_classes={self.num_classes}, "
            f"n_qubits={self.n_qubits}, encoding={self.encoding}, {mode}, "
            f"n_measurements={self.n_measurements}"
        )
