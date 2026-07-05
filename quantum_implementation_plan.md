# Plan: Implement Remaining Survey Techniques + Deeper VQCs

## Context

The survey paper "Toward Quantum-Enhanced Computer Vision" (Connor Druett et al.) covers quantum techniques across five pipeline stages. Most are already implemented in TF-CLIP-quantum, but several remain. This plan maps every unimplemented technique to a concrete implementation, filtered by feasibility on our CPU PennyLane simulator (no D-Wave, no QPU). It also addresses the "lacking quantum parameters" problem — many VQC components have only ~48 trainable circuit weights drowned out by thousands of classical pre/post-net parameters.

Pipeline placement follows the survey's taxonomy strictly:
- **Pre-processing**: raw image operations before ViT backbone
- **Feature Extraction**: how features are learned (backbone-level, temporal pooling)
- **Classification**: decision head
- **Optimisation**: training-time changes
- **Post-processing**: after inference/retrieval

---

## What Already Exists (Do NOT Reimplement)

| Technique | Status | Notes |
|---|---|---|
| QHED | ✅ Pre-processing | raw image edge detection before ViT |
| QTemporal/TQA | ✅ Feature Extraction | data re-uploading over T frames |
| QTD, QGT, QFC | ✅ Feature Extraction | temporal variants |
| QClassifier (VQC head) | ✅ Classification | quantum_layers.py QuantumClassifier |
| QClassifierReupload | ✅ Classification (intractable) | 96-block ~57s/batch — needs redesign |
| QuantumTripletLoss | ✅ Optimisation | kernel metric learning at train time |
| QuantumSwapTest, DurrHoyer | ✅ Post-processing | reranking after retrieval |
| Q-C-Q Interlaced | ✅ Feature Extraction | quantum_interlaced.py |
| Parallel concat (PQCNN-style) | ✅ Feature Extraction | quantum_feature_extractor.py |
| Dense Angle Encoding | ✅ In quantum_layers.py | NOT yet wired to temporal models |
| QChannel, QFrame, QGated, CCG | ✅ Feature Extraction | adapter-style |

---

## Feasibility Filter

**Feasible on CPU PennyLane simulator (implement these):**
Dense angle encoding for temporal models, QPLR, QPCA preprocessing simulation, QuNN-style patch processing, light re-uploading (4–6 blocks), deeper VQC (n_layers=4), SPSA/COBYLA for circuit params

**Hardware-dependent (D-Wave / QPU) — simulation demo only, lower priority:**
Quantum annealing for hyperparams (QAOA), Grover weight search, MRF/Ising annealing

**Not applicable to re-ID:**
QFL (federated), QRL (reinforcement learning), Quantum K-means (unsupervised segmentation), PQWGAN (image generation)

---

## Implementation Plan

### STAGE 1 — Pre-processing (raw image → ViT)

#### 1A. QPCA-Style Quantum Preprocessing (`quantum_models/angle/quantum_pca_preprocess.py`)
In the survey, QPCA is a preprocessing transform on raw image data before the backbone. In our system, raw image data enters as per-patch pixel tensors before ViT's patch embedding. We can't implement real QPCA (requires quantum phase estimation on QPU), but we can implement a **VQC-based channel-wise attention on raw input images** that learns a quantum-enhanced colour/texture filter — the quantum analogue of classical PCA whitening.

Architecture:
```
input image [B*T, C=3, H=256, W=128]
  → global avg pool → [B*T, 3]           (channel descriptor)
  → pre_net: Linear(3 → n_qubits=4)      (3 channels → 4 qubits; small is fast)
  → sigmoid·π → VQC → probs [B*T, 16]   (2^4=16)
  → channel_weights: Linear(16 → 3) + sigmoid  → [B*T, 3] ∈ (0,1)
  → output = input * channel_weights[:, :, None, None]  (channel rescaling)
```

This runs on raw images (pre-processing stage), shapes ViT's input by quantum-learned channel emphasis, is fast (only 3 channels, n_qubits=4), and runs at BOTH train and eval time (unlike adapter-style modules).

- New file: `quantum_models/angle/quantum_pca_preprocess.py`
  - Class `QuantumChannelPreprocess(nn.Module)`
  - Integration point: in `model/make_model_clipreid.py` forward, apply before `image_encoder(img)`
  - Multiplicative residual: `output = image * (1 + channel_weights)` (init to identity)
  - channel_weights linear bias init = 0.0 → sigmoid(0)=0.5 → rescale by 1.5 at init; adjust so init ≈ identity: bias = -4 → sigmoid(-4) ≈ 0.018 → output ≈ input × 1.018 ≈ identity
- New file: `quantum_models/make_model_qpca_pre.py`
- New file: `train_qpca_pre.py`
- Integration pattern: Sequential (quantum-first preprocessing, runs at eval)

#### 1B. QPIE-Style Quantum Image Encoding Filter (`quantum_models/angle/quantum_pie_preprocess.py`)
QPIE encodes pixel intensities as probability amplitudes. In simulation, this means treating each image patch as a quantum state vector and applying a learned unitary rotation. Implementation: apply VQC to each spatial grid cell of the feature map after first conv/patch op.

Practical approach: apply a tiny 2-qubit VQC per 2×2 spatial block on the 3-channel input, producing a quantum-filtered image at the same resolution.

- This is architecturally identical to 1A but applied spatially (per spatial block, not per channel)
- More expensive than 1A but more faithful to QPIE
- New file: `quantum_models/angle/quantum_pie_preprocess.py`
  - Class `QuantumSpatialFilter(nn.Module)`: splits image into 4×4 non-overlapping spatial patches, applies shared 2-qubit VQC to each patch, reassembles
  - n_qubits=2 (very fast: 2^2=4 states), shared weights across all spatial positions
- New file: `train_qpie_pre.py`

---

### STAGE 2 — Feature Extraction

#### 2A. QuNN-Style Temporal Feature Extractor (`quantum_models/angle/quantum_temporal_dense.py`)
In the survey, QuNNs apply quantum circuits as convolutional filters to produce feature maps. In our ViT system, the equivalent is a quantum circuit applied to the per-frame ViT features [B, T, 768] — which is exactly what the existing temporal models do. The survey distinction between "QuNN" and our temporal VQCs is mostly architectural framing; what's genuinely new is **dense angle encoding** which is the information-density improvement that makes QuNNs more expressive: encode 2 features per qubit (RY + RZ), doubling input capacity without more qubits.

Dense angle encoding already exists in `QuantumClassifier` (`quantum_layers.py`) but is NOT wired to the temporal models. Wire it now.

New circuit vs standard:
```python
# Standard AngleEmbedding (1 feature/qubit):
qml.AngleEmbedding(angles[:, :n_q], wires=range(n_q), rotation='Y')
# Dense (2 features/qubit):
qml.AngleEmbedding(angles[:, :n_q], wires=range(n_q), rotation='Y')   # RY
qml.AngleEmbedding(angles[:, n_q:], wires=range(n_q), rotation='Z')   # RZ
# pre_net output: Linear(D → 2*n_qubits) instead of Linear(D → n_qubits)
```

- New file: `quantum_models/angle/quantum_temporal_dense.py`
  - Class `QuantumTemporalDense(nn.Module)` — copy of `quantum_temporal.py` with dense encoding
  - `pre_net: Linear(in_features, 2*n_qubits)` — double the compressed output
  - Circuit adds second `AngleEmbedding(..., rotation='Z')` after first
  - Everything else (residual, upscale, init) identical
- New file: `train_qtemporal_dense.py`
- Apply same pattern to QTD: `quantum_temporal_diff_dense.py` + `train_qtd_dense.py`
- **Why this matters**: directly addresses 768→8 compression bottleneck documented in survey; same circuit depth, no barren plateau increase

#### 2B. Deep VQC Temporal — n_layers=4 with Gradient Clipping
More circuit parameters per block (48→96 weights). Near-identity init tightened to std=0.005. Barren plateau risk managed via gradient clipping at training time.

- New file: `quantum_models/angle/quantum_temporal_deep.py`
  - Identical to `quantum_temporal.py` with default `n_layers=4`, init std=0.005
- New file: `train_qtemporal_deep.py`
  - `--gradient_clip` flag (default 1.0)
  - Logs gradient norm per epoch for barren plateau detection: if grad_norm < 1e-4 flag warning

#### 2C. Light Re-uploading Temporal — Independent Block Weights (`quantum_models/angle/quantum_temporal_reupload.py`)
QClassifierReupload (96 blocks) is intractable (~57s/batch). 4-block variant: ~4–6s/batch. Crucially, each block uses **independent VQC weights** (unlike TQA which shares weights across all T frame uploads). This is the survey's data re-uploading classifier concept applied to temporal features.

- New file: `quantum_models/angle/quantum_temporal_reupload.py`
  - Class `QuantumTemporalReupload(nn.Module)`
  - `n_reupload` independent `StronglyEntanglingLayers` weight tensors (nn.ParameterList)
  - Each block: `AngleEmbedding(frame_angles_t) → SEL_k` — frame index t cycles modulo n_reupload
  - After all T frames: measure probs → upscale → residual
  - Total circuit params: n_reupload × n_layers × n_qubits × 3 = 4×2×8×3 = 192 (vs 48 for TQA)
- New file: `train_qtemporal_reupload.py` (`--n_reupload` arg, default 4)

#### 2D. QPCA Autoencoder on Extracted Features (`quantum_models/angle/quantum_autoencoder.py`)
After ViT extracts per-frame features [B, T, 768], a VQC autoencoder compresses and reconstructs them. This is "quantum autoencoder as feature extractor" from the survey. The VQC bottleneck IS the compressed representation; reconstruction loss forces it to preserve information. At inference, only the encoder path runs.

Architecture:
```
x [B, 768] → pre_net(768→n_q) → sigmoid·π → VQC → probs [2^n_q]  ← quantum bottleneck
           ↓                                                         ← reconstruct
           upscale(2^n_q→768)  → x_recon [B, 768]
Additional loss: MSE(x_recon, x.detach()) * recon_weight
```

- New file: `quantum_models/angle/quantum_autoencoder.py`
  - Class `QuantumAutoEncoder(nn.Module)` integrating into temporal pooling (operates on mean-pooled [B,768])
- New file: `loss/quantum_recon_loss.py` — `recon_weight * F.mse_loss(x_recon, x_original)` added to total loss
- New file: `train_qautoencoder.py` (`--recon_weight` arg, default 0.1)

---

### STAGE 3 — Classification

#### 3A. Multi-Head VQC Classifier (`quantum_models/angle/quantum_multihead_classifier.py`)
Current single-head QClassifier: 768→n_q→VQC→256→1604. Problem: single 256-dim VQC output is a severe bottleneck for 1604 classes. Multi-head: K=4 independent VQC heads, each producing 256-dim quantum features, concatenated to 1024-dim, then projected to 1604 classes. Total circuit params: 4×48=192.

- New file: `quantum_models/angle/quantum_multihead_classifier.py`
  - Class `QuantumMultiHeadClassifier(nn.Module)`
  - K `QuantumClassifier` instances (reuse from `quantum_layers.py`)
  - Each head: separate `pre_net(768→n_q)` + independent VQC weights
  - Concat outputs: `[B, K*256]` → `Linear(K*256, num_classes)`
  - `bypass_quantum=True`: replaces each head with `Linear(768→256)+ReLU`
- New file: `train_qmultihead.py` (`--n_heads` arg, default 4)

#### 3B. Deep QClassifier (n_qubits=10, n_layers=4)
n_qubits=10 → 2^10=1024-dim VQC output (vs 256 for n_qubits=8), much richer for 1604 classes. The existing `QuantumClassifier` in `quantum_layers.py` supports this with just hyperparameter changes.

- No new module file — modify `quantum_models/make_model_qclassifier.py` to accept `--n_qubits 10 --n_layers 4`
- Modify `train_qclassifier.py` to expose these args
- New training script `train_qclassifier_deep.py` with these as defaults + gradient clipping

---

### STAGE 4 — Optimisation

#### 4A. SPSA Hybrid Optimizer (`utils/hybrid_optimizer.py`)
Survey finding: SPSA and COBYLA are more noise-robust than parameter-shift for deep circuits. Hybrid: Adam for classical params (pre_net, upscale, gate_net), SPSA for quantum circuit weights only (`q_weights` / `qlayer.weights`). SPSA gradient estimate uses two forward passes with simultaneous random perturbation — cost independent of parameter count.

- New file: `utils/hybrid_optimizer.py`
  - Class `SPSAOptimizer(torch.optim.Optimizer)`
    - `__init__(params, lr=0.01, c=0.01, perturbation='rademacher')` — standard SPSA
    - `step(closure)`: two forward passes at (θ+c·Δ) and (θ-c·Δ), estimate gradient via finite diff
  - Function `make_hybrid_optimizer(model, cfg)`: splits model.parameters() by name — `'q_weights'` or `'qlayer.weights'` → SPSA; all others → Adam
- New file: `train_qtemporal_spsa.py` — uses hybrid optimizer
- `q_weights` naming is consistent across all quantum modules already

#### 4B. Entanglement Pattern Search (`quantum_models/angle/quantum_temporal_configurable.py`)
Lightweight version of DQAS: compare different entanglement topologies in StronglyEntanglingLayers. `'full'` (default) vs `'ring'` (nearest-neighbour only) vs `'linear'` (chain). Ring/linear are less expressive but more hardware-friendly and potentially less barren-plateau-prone.

- New file: `quantum_models/angle/quantum_temporal_configurable.py`
  - Class `QuantumTemporalConfigurable(nn.Module)`
  - `entanglement` arg: `'full'` | `'ring'` | `'linear'`
  - `'ring'` and `'linear'` implemented via explicit `qml.CNOT` loops replacing `StronglyEntanglingLayers`
- New file: `train_qtemporal_ent.py` (`--entanglement` arg for sweeping)

---

### STAGE 5 — Post-processing

#### 5A. QPLR — Quantum Probabilistic Label Refining (`loss/quantum_label_refiner.py`)
Most novel unimplemented technique. After the classifier produces logits [B, 1604], a VQC processes them to produce soft labels capturing inter-class quantum correlations. Particularly relevant for AG-VPReID: aerial/ground pairs create inherent class ambiguity (same identity looks different across cameras). Standard cross-entropy treats all wrong classes identically; QPLR learns quantum correlations between visually similar classes.

Architecture:
```
logits [B, 1604]
  → top-K selector (K=32) → [B, 32]          (top-32 class logits only; tractable)
  → pre_net: Linear(32 → n_qubits=8)
  → sigmoid·π → VQC → probs [B, 256]
  → refine_net: Linear(256 → 32) + softmax    → quantum-refined soft weights for top-32 classes
  → scatter back to full [B, 1604]             (non-top-32 get logit-proportional weight)
  → KL divergence against one-hot target
```

- New file: `loss/quantum_label_refiner.py`
  - Class `QuantumLabelRefiner(nn.Module)`
  - `forward(logits, target)` → `qplr_loss` (KL div)
  - `kl_weight` to blend with standard cross-entropy: `total = CE + kl_weight * KL`
  - Top-K selection is non-differentiable; use straight-through for gradients through selector
- Modify `loss/make_loss.py` to add `make_loss_qplr()` function
- New file: `train_qplr.py` (`--top_k_classes 32 --kl_weight 0.5`)

#### 5B. Zero Noise Extrapolation — ZNE (`utils/noise_mitigation.py`)
Required for hardware deployment. On `default.qubit` simulator it has no effect (no noise), but demonstrates readiness. Implements ZNE by intentionally adding `qml.DepolarizingChannel` noise at scaled levels and extrapolating to zero noise via Richardson extrapolation.

- New file: `utils/noise_mitigation.py`
  - Function `apply_zne(circuit_fn, inputs, weights, noise_factors=[1, 3], noise_prob=0.01)`
  - Evaluates circuit with DepolarizingChannel at each noise_factor, Richardson extrapolates
  - Wrapper class `ZNECircuit` wrapping any `@qml.qnode`
- Add `--use_zne` flag to eval scripts (eval_agvpreid_quantum.py)
- Useful when moving to hardware

---

## "More VQC Parameters" Summary

| Component | Current VQC params | Proposed | New VQC params |
|---|---|---|---|
| QTemporal (standard) | 48 (2×8×3) | Dense angle encoding | 48 params, but encodes 16 features (vs 8) |
| QTemporal | 48 | n_layers=4 deep | 96 (4×8×3) |
| QTemporal | 48 | Light re-uploading n=4 blocks | 192 (4×2×8×3) |
| QClassifier | 48 | n_qubits=10, n_layers=4 | 120 (4×10×3) |
| QClassifier | single head 48 | Multi-head K=4 | 192 (4×48) |

---

## Critical Files to Reference

- `quantum_models/angle/quantum_layers.py` — `QuantumClassifier` base VQC block (dense_angle already here); CPU-pinning pattern; init conventions
- `quantum_models/angle/quantum_temporal.py` — canonical temporal VQC pattern (copy-base for 2A, 2B, 2C)
- `quantum_models/angle/quantum_temporal_gated.py` — gate mechanism and `last_gates` pattern
- `quantum_models/angle/quantum_feature_extractor.py` — parallel concat pattern (reference for multi-head)
- `loss/make_loss.py` — how to register new loss variants (see `make_loss_q_triplet` lines 101-182)
- `loss/quantum_triplet_loss.py` — VQC-in-loss pattern (QPLR follows same structure)

---

## Recommended Build Order

1. **Dense angle encoding temporal** (2A) — smallest change, directly addresses documented bottleneck
2. **QPLR** (5A) — novel post-processing contribution, aerial/ground ambiguity angle
3. **Light re-uploading** (2C) — fixes the intractable QClassifierReupload
4. **Deep VQC with gradient clipping** (2B)
5. **QPCA channel preprocessing** (1A) — quantum pre-processing stage demo
6. **QPIE spatial filter** (1B)
7. **QPCA autoencoder** (2D)
8. **SPSA hybrid optimizer** (4A)
9. **Multi-head classifier** (3A), Deep QClassifier (3B)
10. **Entanglement structure search** (4B), ZNE (5B) — analysis/hardware-readiness

---

## Verification

For each new module:
1. Unit test: instantiate, pass dummy `[4, 8, 768]` or `[4, 768]`, check output shape
2. 5-epoch smoke test: `SOLVER.STAGE2.MAX_EPOCHS 5 SOLVER.STAGE2.EVAL_PERIOD 999`
3. Compare train `acc_id1` at ep5 vs classical baseline (~0.073 at ep5)
4. Full 40ep eval for best performers against classical ep40 (58.1% Rank-1)
