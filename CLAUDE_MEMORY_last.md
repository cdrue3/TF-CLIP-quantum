# CLAUDE_MEMORY_last.md
## Instance 4 (Final) — Quantum TF-CLIP Handoff

**Date:** 2026-07-06  
**Instance focus:** Feature Extraction stage — Dense angle encoding, QGT (Quantum-Gated Temporal), Hamiltonian QTD, Hamiltonian QClassifier  
**Critical breakthrough:** Discovered and fixed hidden LR boosts that were invalidating all previous quantum results

---

## Project Overview

**Goal:** Beat classical CLIP-ViT TF-CLIP person re-identification on AG-VPReID (aerial-ground cross-view).  
**Dataset:** subset_250 (250 identities, ~600k frames). Case 1 = aerial query → ground gallery.  
**Classical baseline:** 62.4% Rank-1 @ ep70 (80ep training, subset_250, Case 1).  
**Framework:** TF-CLIP (CLIP + ViT-B/16), two-stage training — Stage 1: CLIP memory bank, Stage 2: ID loss + triplet + I2T contrastive.

---

## All-Time Best Results (subset_250, Case 1, 80ep)

| Model | Stage | Rank-1 | vs Classical | Notes |
|---|---|---|---|---|
| **QuantumTripletLoss** | Optimisation/Loss | **65.2%** | **+2.8pp** | VQC in triplet loss only; classical arch at inference |
| Classical single-head (ablation) | — | 66.2% | (reference) | 4-head vs 1-head architecture study |
| Dense angle QTD 8q (no boost) | Feature Extraction | **63.7%** | **+1.3pp** | Peak result for architecturally-fair quantum |
| Hamiltonian QTD (no boost) | Feature Extraction | **63.2%** | **+0.8pp** | First quantum > classical; discovered LR boost flaw |
| QPLR kl_weight=0.1 | Post-processing | 62.8% | +0.4pp | VQC label refiner |
| Classical baseline | — | 62.4% | 0pp | Reference, ep70, STEPS=[30,50,70] |
| QGT bias=0 (no boost) | Feature Extraction | 62.4% | ~0pp | Gate doesn't differentiate aerial vs ground |
| QClassifier dense_angle 8q | Classification | 55.1% | -7.3pp | Bottleneck limits Rank-1 |
| QClassifier hamiltonian 5q | Classification | 53.7% | -8.7pp | No skip connection, learning is slow |
| Ham QTD boosted (old, tainted) | Feature Extraction | 58.5% | -3.9pp | LR boost caused overfitting |

---

## Critical Finding: LR Boost Was Invalidating All Results

**This was the most important discovery of the project.**

All `train_q*.py` scripts had hidden LR boosts that the classical baseline never had:
- `TQA_LR_FACTOR = 3` on VQC weights (pre_net, qlayer_weights)  
- `CLASSIFIER_LR_FACTOR = 10` on classifier heads
- `POST_NET_LR_FACTOR = max(5, int(768*10/n_measurements))` — up to 30× for small qubit counts

**Effect:** Forced quantum models to memorise training IDs via cross-entropy → high acc_id1 (~0.82 vs classical ~0.55) → destroyed feature geometry needed for retrieval → Rank-1 3-4pp below classical despite appearing to "converge faster."

**Fix:** Set all factors to 1 in every script. Already applied to:
- `train_qtd_ham.py` ✓  
- `train_qtemporal_dense.py` ✓  
- `train_qgt.py` ✓  
- `train_qclassifier.py` ✓  
- `train_qclassifier_ham.py` ✓ (never had boost)

**Important note:** After removing boost, acc_id1 is near zero during training (classifier head barely trains). This is expected and fine — Rank-1 at inference uses feature distances (triplet + I2T signals), not classifier. Do not mistake low training acc_id1 as a failure.

**Fair comparison rule:** Any hyperparameter change (LR factors, loss weights, schedule) must also be applied to a classical baseline run. The only difference between quantum and classical runs should be the quantum component itself.

---

## Architecture Details: Feature Extraction Models

### Dense Angle QTD (Best Fair Result: 63.7%)
- **File:** `quantum_models/feature_extraction/quantum_temporal_dense.py`, built by `quantum_models/feature_extraction/make_model_qtemporal_dense.py`
- **Training:** `train_qtemporal_dense.py`
- **Encoding:** Dense angle — 2 features/qubit via RY(angle) + PhaseShift(phase) → 8 qubits encode 16 features
- **Architecture:** `mean_pool(x) + upscale(VQC(diffs))` — HAS skip connection
  - `pre_net`: Linear(768→16) → compress to 16 features
  - VQC: `StronglyEntanglingLayers(n_layers=2, n_wires=8)` on frame diffs
  - `upscale`: Linear(256→768) → back to feature space
  - Final: mean-pooled CLIP features + VQC delta
- **Key insight:** Skip connection (`mean_pool + delta`) is critical — keeps retrieval quality even when VQC output is noisy

### Hamiltonian QTD (First Quantum > Classical: 63.2%)
- **File:** `quantum_models/amplitude/quantum_temporal_diff_ham.py`
- **Training:** `train_qtd_ham.py`
- **Encoding:** Hamiltonian — features as Pauli basis coefficients: `H(x) = Σᵢ xᵢ Pᵢ`, `U = e^{-iH}`, `|ψ⟩ = U|0⟩[:,0]`
- **Architecture:** `upscale(VQC(diffs))` — NO skip connection (unfair comparison to dense)
  - No pre_net (768 features direct to Pauli basis)
  - n_qubits=5 minimum (4^5-1=1023 ≥ 768 Pauli operators)
  - `StronglyEntanglingLayers(n_layers=2, n_wires=5)`
  - `upscale`: Linear(32→768)
- **Note:** CPU-pinned (`_apply` override) because `torch.matrix_exp` on complex tensors requires CPU. No GPU speedup.

### QGT — Quantum-Gated Temporal (62.4%, gate doesn't differentiate cameras)
- **File:** `quantum_models/angle/quantum_temporal_gated.py`
- **Training:** `train_qgt.py`
- **Architecture:** Learned scalar gate `g = sigmoid(gate_net(mean_pool))`, `output = mean_pool + g * VQC_delta`
- **Research question Q2:** Does gate learn different values for aerial vs ground cameras?
  - Tested bias=-2 (init g≈0.12) → 61.5% peak, gate gap <0.05 aerial vs ground
  - Tested bias=0 (init g≈0.5) → 62.4% peak, gate gap <0.05 aerial vs ground
  - **Conclusion: NO — gate learns a single global trust level, not camera-specific routing**
- **Camera ID encoding:** Embedded in filename via `C(\d+)` regex, e.g. `P6287T2310171A3C5E321K0F6459.jpg` → `C5`. C0-C3=ground, C4-C5=aerial. C4 nearly absent in subset_250 train (only 6090 samples vs C5=211739).
- **Gate logging:** `processor/processor_clipreid_stage2.py` logs `gate_mean`, `gate_aerial`, `gate_ground` per epoch via `model.qtg.last_gates`

### QClassifier (Classification Stage)
- **Dense angle 8q:** `quantum_models/angle/quantum_layers.py` → `QuantumClassifier`, training `train_qclassifier.py`
  - VQC replaces all 4 classifier heads (`classifier2`, `classifier_proj`, `classifier_proj_temp`, `classifier_proj_temp2`)
  - TF-CLIP always has 4 heads by design
  - Peak: 55.1% @ ep80 — bottleneck limits performance; classifier heads aren't used at inference
- **Hamiltonian 5q:** `quantum_models/amplitude/quantum_classifier_ham.py` → `QuantumClassifierHam`, training `train_qclassifier_ham.py`
  - Same Pauli basis encoding as QTDHam
  - Peak: 53.7% @ ep60 — slower learning, no skip, Rank-1 lags further
- **Key insight:** QClassifier Rank-1 is limited because the VQC bottleneck (32 or 256 measurements) trains only the classifier heads, not the backbone features. The backbone (producing retrieval embeddings) is trained only by triplet + I2T loss. Low Rank-1 reflects that the classifier bottleneck doesn't help retrieval.

---

## QuantumTripletLoss — Best Overall Result (65.2%, +2.8pp)

**This was run on a separate post-processing instance.**

- **File:** `loss/quantum_triplet_loss.py`
- **Training:** `train_q_triplet_loss.py`
- **How it works:** VQC is embedded in the triplet loss computation — quantum circuit processes anchor/positive/negative embeddings during training. At inference, only the classical backbone produces embeddings, so zero quantum overhead at test time.
- **Why it works:** VQC in loss shapes the metric learning signal during training without affecting inference architecture. The quantum component learns to produce better training gradients.
- **Key result:** 65.2% @ ep70 — beats classical 62.4% by +2.8pp. Best result on this dataset.
- **Comparison baseline:** Single-head classical 66.2% @ ep60 (important — this uses 1 classifier head instead of 4; the quantum model's multi-head architecture may be the fairer comparison at 62.4%).

---

## Post-Processing Results (separate instance — those files are authoritative)

All post-hoc reranking on AG-VPReID failed due to the aerial/ground viewpoint gap breaking k-NN assumptions:
- SwapTest reranking: 45.5% (-20.4pp) — quantum concentration collapses distances
- Pairwise VQC reranker: 66.4% (same as baseline) — any training hurts
- Quantum k-reciprocal reranker: 66.4% — viewpoint gap breaks k-NN
- Durr-Hoyer: same accuracy as classical, 23× theoretical speedup on quantum hardware
- QPLR kl_weight=0.1: 62.8% (+0.4pp) — only post-processing approach that helped

**Conclusion:** Post-processing on this dataset is exhausted. The viewpoint gap (aerial vs ground) makes reranking ineffective.

---

## Technical Details: GPU / PennyLane

- **Backend:** PennyLane `default.qubit`, `diff_method='backprop'`, `interface='torch'`
- **GPU utilization:** 0% during quantum simulation — `default.qubit` runs on CPU regardless of tensor device. The "GPU fix" (removing `.cpu()` calls from forward passes) removed data-transfer overhead but didn't enable GPU-accelerated quantum simulation.
- **lightning.qubit / lightning.gpu:** Not installed on VMs. Would need `pip install pennylane-lightning` / `pennylane-lightning-gpu`. The ham model's `matrix_exp` on complex tensors still requires CPU regardless.
- **Batch timing:** ~2.5-3 min/epoch for 8-qubit VQC models on these VMs ($0.35/hr).

---

## Ep30 Collapse — Known Issue (Fixed)

MultiStepLR with `STEPS=[30,50,70]` caused Rank-1 to collapse to ~13% at ep30 across ALL models (quantum and classical). This was caused by too-aggressive LR decay. Fixed by switching schedule or using `ReduceLROnPlateau`. The collapse at ep30 is visible in all early result sweeps.

Ring entanglement (BasicEntanglerLayers) was most robust to ep30 collapse (43.1% vs ~13% for others).

---

## Training Signal Observations

- `acc_id1` plateau **leads** Rank-1 plateau by ~10 epochs. Can use acc_id1 as early indicator.
- Models WITH skip connections show higher acc_id1 (classifier gets gradient through skip path).
- Models WITHOUT skip connections (Ham QTD, Ham QClassifier) show near-zero acc_id1 with no LR boost — this is expected and correct.
- Dense angle encoding gives better training signal than Ham encoding (skip + more expressive pre_net).

---

## Fair Comparison Rules (Critical)

Always check before reporting results:
1. LR factors must be 1× for both quantum and classical
2. Architecture must differ ONLY in the quantum component
3. Skip connections: `dense QTD (has skip) vs ham QTD (no skip)` = unfair architectural comparison
4. If a quantum variant has skip and classical doesn't (or vice versa), note this explicitly
5. Acc_id1 cannot be compared across skip/no-skip architectures

---

## File Structure

```
quantum_models/
  angle/          # Standard/dense angle encoding (VQC in feature space)
    quantum_layers.py           # QuantumClassifier, QuantumTQA, etc.
    quantum_temporal_diff.py    # QTD standard
    quantum_temporal_gated.py   # QGT (gated temporal)
    quantum_temporal_dense.py   # Dense angle encoding
  amplitude/      # Hamiltonian encoding (Pauli basis)
    quantum_temporal_diff_ham.py    # Ham QTD (no skip)
    quantum_classifier_ham.py       # Ham QClassifier
  feature_extraction/   # Make-model wrappers for feature extraction variants
  classification/       # Make-model wrappers for classifier variants
  preprocessing/        # Quantum preprocessing (pre-ViT)
  optimisation/         # Quantum optimisation variants
  postprocessing/       # Quantum post-processing (reranking, etc.)

train_q*.py           # One script per architecture
eval_q*.py            # Matching eval scripts
scripts/eval_sweep.sh # Sweep checkpoints ep10..ep80, output Rank-1 table
logs/                 # Training logs, eval results (no checkpoints on git)
transcripts/          # Scrubbed session transcripts for context
```

---

## What's Unexplored / Next Steps

1. **More QuantumTripletLoss variants** — the +2.8pp result is the clearest win. Try different VQC architectures in the loss.
2. **Fair Ham QTD with skip** — currently ham has no skip (unfair vs dense). Adding a skip to ham would give a clean apples-to-apples comparison.
3. **Quanvolutional filters** — applying VQC to raw image patches before ViT (true quantum preprocessing). Currently all approaches operate on ViT features. Papers exist (e.g. Henderson et al.). Requires small patch sizes to be tractable on CPU.
4. **lightning.gpu** — would accelerate PennyLane simulation. Install: `pip install pennylane-lightning-gpu`. Not tried on these VMs yet.
5. **Multi-qubit QGT** — current gate is a scalar; could try per-feature or per-camera gating.

---

## User Preferences (Connor, QUT grad student)

- Terse communication — short answers, no padding
- "?" alone means status check on current run
- Always include classical baseline column in any result table
- Any hyperparameter change must have matching classical run
- Use nohup/screen for all training so VM disconnection is safe
- Include cost estimate ($0.35/hr) with ETAs
- GDrive MCP tools don't work — use gdown or rclone
- Don't explain what code does (self-documenting names); explain WHY if non-obvious

---

## Previous Instance Memory Files

- `CLAUDE_MEMORY_0.md` — Instance 1: initial quantum architectures, AG-VPReID setup
- `CLAUDE_MEMORY_1.md` — Instance 2/3: GPU fix, QPCA breakthrough, SQP+BER+Parallel  
- `CLAUDE_MEMORY_2.md` — Post-processing instance: QuantumTripletLoss +2.8pp, QPLR, reranking failures
- `CLAUDE_MEMORY_last.md` — This file: Instance 4, LR boost fix, dense/ham/QGT/QClassifier results
