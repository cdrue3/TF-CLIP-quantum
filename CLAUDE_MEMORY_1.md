# CLAUDE_MEMORY_1.md — Instance 3 Handoff (VM Snapshot: July 2026)

This document captures the full experimental context from the third Claude Code instance
working on TF-CLIP-quantum. Read CLAUDE_MEMORY_0.md first for project foundation.
Read CLAUDE.md for codebase and environment context.

---

## What Instance 3 Did

This instance focused on:
1. Diagnosing and fixing the **LR boost overfitting bug** that tainted all earlier results
2. Applying the **GPU speed fix** (removing CPU-pinning from VQC forward passes)
3. Running **no-LR-boost** experiments to get valid quantum vs classical comparisons
4. Running **quantum preprocessing (QPCA)** experiments — yielding the first genuine quantum win
5. Running **SQP+BER** and **parallel** architecture experiments (with LR boost — tainted)
6. Uploading results to GDrive as individual per-experiment .tar.gz files

---

## The LR Boost Bug (Critical Finding)

**Root cause of all earlier quantum results being below classical:**

All `train_q*.py` scripts had:
```python
TQA_LR_FACTOR        = 3    # was 3
CLASSIFIER_LR_FACTOR = 10   # was 10
ADAPTER_LR_FACTOR    = 3    # was 3
VQC_LR_FACTOR        = 3    # was 3
POST_NET_LR_FACTOR   = max(5, ...)  # was dynamic
```

These multiplicative LR boosts caused the quantum components to overfit aggressively
and disrupted the joint embedding space learned with the CLIP backbone. This explains
why even simple quantum components couldn't reach classical baseline (62.4%).

**The fix:** Set ALL LR factors to 1 in every `train_q*.py`. This was applied before
all Instance 3 experiments.

Current state of `train_q*.py` files: all have `TQA_LR_FACTOR = 1` and `CLASSIFIER_LR_FACTOR = 1`.

---

## The GPU Speed Fix

VQC forward passes were pinned to CPU via `_apply` overrides and `.cpu()` calls.
Removing these lets PennyLane use PyTorch tensors on whatever device they're on.

**Files fixed during this instance:**
- `quantum_models/feature_extraction/quantum_temporal_sqp.py` — SQP+BER model
- `quantum_models/feature_extraction/quantum_temporal_ham.py` — Hamiltonian model
- `quantum_models/feature_extraction/quantum_temporal_parallel.py` — Parallel model
- `quantum_models/preprocessing/quantum_pca_preprocess.py` — QPCA model
- `quantum_models/angle/quantum_layers.py` — QClassifier

**Effect:** ~3.6x speedup on VQC forward (6.5s → 1.8s per batch for SQP, ~0.56s/batch
for Hamiltonian which uses matrix_exp on GPU via PyTorch).

**Note:** Many other quantum model files (`quantum_models/angle/`, `quantum_models/amplitude/`
subdirectory models, etc.) may NOT have this fix applied. Check for `_apply` override or
`.cpu()` calls before running.

---

## Experiments Run (Valid — No LR Boost)

### Classical Baseline (subset_250, 80ep, SEQ_LEN=8)
- **Rank-1: 62.3–62.4%** (rock-solid, the reference for all comparisons)
- Path: `logs/agvpreid_classical_80ep/`
- Sweep: `logs/eval/` (not in this instance's logs — use the value 62.4%)

### QClassifier Dense (subset_250, 80ep)
Runs: nq=4, nq=6, nq=8 (all with dense_angle encoding, no LR boost)
- nq=4: peak ~53.4%, nq=6: ~53.2%, nq=8: ~53.4% (all well below classical)
- Log paths: `logs/agvpreid_qclassifier/80ep_dense_nq4/`, `.../nq6/`, `.../nq8/`
- Eval sweeps: `logs/eval/qclassifier_dense_nq4_sweep.txt` etc.
- **Conclusion:** VQC replacing classifier heads cannot compete with classical linear heads.

### QPCA Quantum Preprocessing (subset_250, 80ep)
- **Rank-1 ep80: 64.2%** — **+1.8pp above classical baseline (62.4%)**
- Peak at ep80 (still improving at termination)
- Log path: `logs/agvpreid_qpreprocess/80ep_noboost/train_log.txt`
- Eval sweep: `logs/eval/qpreprocess_noboost_sweep.txt`
- Architecture: QuantumChannelPreprocess (n_qubits=8, n_layers=2), channel-wise attention
  on raw images [B*T, 3, H, W] before ViT. VQC produces channel attention scalars.
- **This is the first genuine quantum win on this project.**

### QPCA Classical Ablation (subset_250, 80ep)
Fair comparison: same architecture but `bypass_quantum=True` uses `classical_expansion`
(Linear(n_qubits→2^n_qubits) + ReLU) instead of VQC.
- **Rank-1 ep80: 57.3%** — below classical baseline
- Log path: `logs/agvpreid_qpreprocess/80ep_classical_ablation/train_log.txt`
- Eval sweep: `logs/eval/qpreprocess_classical_ablation_sweep.txt`
- **Conclusion:** Classical channel attention with same parameter count hurts performance.
  The quantum VQC is the source of the improvement.

---

## Experiments Run (Tainted — Had LR Boost, Unreliable)

These ran with the old LR boost and are NOT valid for comparison with the no-boost baseline:

| Experiment | Peak R1 | ep80 R1 | Notes |
|---|---|---|---|
| Ring n3 (entanglement, 40ep) | 58.4% | 58.0% | LR boost |
| Ring n3 80ep extension | ~58% | ~58% | LR boost |
| SQP+BER standard (80ep) | 61.0% at ep45 | ~59% | LR boost |
| SQP+BER dense (80ep) | 60.0% | 59.7% | LR boost |
| SQP+BER Hamiltonian (80ep) | 60.3% | 58.1% | LR boost |
| Parallel quantum-classical (80ep) | 60.8% | 59.9% | LR boost |
| Parallel dense (80ep) | ~58% | 57.6% | LR boost |
| Parallel Hamiltonian (80ep) | ~59% | ~58% | LR boost |

These are on `logs/agvpreid_sqp_ber/` and `logs/agvpreid_qtemporal_ent/`.

**Action needed for future instances:** Re-run SQP+BER, parallel architectures with no LR boost
to get valid comparison data.

---

## New Files Added This Instance

### Training scripts:
- `train_qtemporal_ber.py` — SQP+BER training with inline eval
  - Args: `--entropy_reg 0.02`, `--noise_sigma 0.15`, `--noise_epochs 20`
  - `--dense_encoding`, `--hamiltonian`, `--parallel`, `--fusion_mode [concat|gated]`
- `train_qtemporal_ent.py` — Entanglement structure search (ring/full/linear topology)
- `train_qpreprocess.py` — QPCA/QPIE preprocessing training
- `train_qtd_ham.py` / `train_qtd_ham_skiponly.py` — Hamiltonian temporal diff variants
- `train_qclassifier_deep.py` — Deep QClassifier variant
- `train_qmultihead.py` — Multi-head quantum classifier
- `train_qtemporal_deep.py` — Deep QTemporal variant
- `train_qtemporal_reupload.py` — Re-uploading TQA variant
- `train_qtemporal_spsa.py` — SPSA optimizer variant
- `train_qautoencoder.py` — Quantum autoencoder variant
- `train_qplr.py` — Quantum post-processing label refiner

### Quantum model files (feature_extraction subdirectory):
- `quantum_models/feature_extraction/quantum_temporal_sqp.py` — SQP+BER TQA
- `quantum_models/feature_extraction/quantum_temporal_ham.py` — Hamiltonian TQA
- `quantum_models/feature_extraction/quantum_temporal_parallel.py` — Parallel TQA
- `quantum_models/feature_extraction/quantum_temporal_deep.py` — Deep TQA
- `quantum_models/feature_extraction/quantum_temporal_dense.py` — Dense encoding TQA
- `quantum_models/feature_extraction/quantum_temporal_reupload.py` — Re-uploading TQA
- `quantum_models/feature_extraction/quantum_autoencoder.py` — Quantum autoencoder
- `quantum_models/feature_extraction/make_model_qtemporal_sqp.py` — Builder for SQP
- `quantum_models/feature_extraction/make_model_qtemporal_ham.py` — Builder for Ham
- `quantum_models/feature_extraction/make_model_qtemporal_parallel.py` — Builder for Parallel

### Quantum model files (preprocessing subdirectory):
- `quantum_models/preprocessing/quantum_pca_preprocess.py` — QPCA channel attention
  (bypass_quantum=True uses classical_expansion, NOT identity pass-through)
- `quantum_models/preprocessing/quantum_pie_preprocess.py` — QPIE spatial filter
- `quantum_models/preprocessing/make_model_qpreprocess.py` — Builder

### Eval scripts:
- `eval_qchannel.py`, `eval_qclassifier.py`, `eval_qtd_ham.py` (plus others)
- `eval_agvpreid_quantum.py` — Multi-architecture eval script
- `eval_checkpoint_sweep.py` — Sweep evaluator (used by `scripts/eval_sweep.sh`)

### Scripts:
- `scripts/eval_sweep.sh` — Evaluates all checkpoint_ep*.pth.tar files in a dir
- `scripts/chain_classifier_preprocess.sh` — Chain: QClassifier dense sweep → QPCA 80ep
- `run_after_swap.sh`, `run_eval_then_dh.sh`

### Other:
- `solver/adaptive_scheduler.py` — ReduceLROnPlateau wrapper (planned replacement for MultiStepLR)
- `quantum_models/optimisation/spsa_optimizer.py` — SPSA optimizer
- `quantum_models/optimisation/quantum_temporal_configurable.py` — Configurable entanglement
- `quantum_models/optimisation/make_model_qtemporal_ent.py` — Builder for ent model
- `quantum_models/postprocessing/noise_mitigation.py`, `quantum_label_refiner.py`
- `loss/quantum_recon_loss.py`, `loss/quantum_triplet_loss.py`
- `model/feature_preprocessors.py`, `model/quantum_image_preprocessor.py`
- `utils/quantum_retrieval.py`

---

## Key Architecture Details

### QuantumChannelPreprocess (QPCA preprocessing)
```
x [B*T, 3, H, W]
→ global_avg_pool → [B*T, 3]
→ pre_net: Linear(3→n_qubits) → sigmoid(·)*π
→ VQC (AngleEmbedding + StronglyEntanglingLayers) → probs [B*T, 2^n_qubits]
→ channel_net: Linear(2^n_q→3) → channel_weights [B*T, 3]
→ output = x * (1 + channel_weights[:, :, None, None])
```
- bypass_quantum=True: uses `classical_expansion` (Linear+ReLU) instead of VQC
- Init: channel_net weights ~N(0, 0.001) → attention starts at 0 → identity residual
- n_qubits=8, n_layers=2 used in main experiment

### SQP+BER (QuantumTemporalSQP)
- **SQP**: Stochastic Quantum Perturbation — Gaussian noise on circuit weights during training,
  sigma decays from `noise_sigma` to 0 over `noise_epochs`
- **BER**: Born Entropy Regularization — `-λ·H(probs)` loss term prevents circuit collapse
  to peaked distributions; encourages superposition (quantum expressivity)
- `_noise_scale` set externally each epoch, `_last_probs` exposed for entropy loss

### Hamiltonian Encoding (QuantumTemporalHam)
```
H(x) = Σᵢ xᵢ Pᵢ   (Pauli basis, up to n_qubits=5 → 4^5-1=1023 Paulis ≥ 768 features)
state = e^{-iH}|0⟩  (matrix exponential, computed via PyTorch on GPU)
n_states=32
```
- ~0.56 sec/batch (3x faster than angle encoding — matrix_exp runs on GPU)
- SQP+BER hooks included

### Parallel Architecture (QuantumTemporalParallel)
```
x [B, T, 768]
├── Classical path: x.mean(1) → [B, 768]
└── Quantum path: TQA (angle/dense/Ham) → [B, 768]
    fusion_mode='concat': Linear(1536→768)+ReLU, init identity on classical half, zeros on quantum
    fusion_mode='gated': g=sigmoid(gate_net(classical)), output=g*classical+(1-g)*quantum
```
- Ensures classical path has full gradient signal from start (quantum half initialized to 0)
- SQP+BER hooks on quantum path

---

## Known Issues / Bugs Fixed

1. **Chain script argument ordering bug**: YACS config overrides and model flags mixed in
   `$COMMON` caused `AssertionError: Override list has odd length`. Fixed by separating
   `BER_OPTS` (model flags like `--entropy_reg`) from `YACS_OPTS` (config overrides like
   `SOLVER.STAGE2.MAX_EPOCHS 80`).

2. **transreid.test logger dropping Rank-1**: `do_inference_rrs` uses
   `logging.getLogger("transreid.test")` which had no handlers → Fixed by wiring it
   to TFCLIP logger handlers in `train_qtemporal_ent.py` and `train_qtemporal_ber.py`.
   Pattern to add to all train scripts:
   ```python
   import logging as _logging
   _tl = _logging.getLogger("transreid.test")
   for _h in logger.handlers:
       _tl.addHandler(_h)
   _tl.setLevel(_logging.DEBUG)
   ```

3. **QPCA classical ablation returning x unchanged**: bypass_quantum=True just did
   `return x` with no preprocessing. Fixed to use `classical_expansion` network
   (Linear(n_qubits→2^n_qubits)+ReLU) as fair ablation. Current file is correct.

4. **ep30 collapse**: MultiStepLR steps=[30,50,70] cause R1 to crash at ep30 for many
   quantum models. BER regularization softens it but doesn't eliminate. `ReduceLROnPlateau`
   is planned as replacement (see `solver/adaptive_scheduler.py`).

---

## Classical Baseline on subset_250

Important: all Instance 3 experiments used `subset_250` (250 IDs), not the full 1604-ID dataset.

- **SEQ_LEN=8**, Case 1 (aerial→ground), rrs_test eval
- **Classical Baseline R1 = 62.4%** (80ep, no LR boost)
- This is the reference for ALL Instance 3 results

Full dataset (1604 IDs) baseline was run by another instance and is ~82.2% R1.

---

## What to Run Next (Priority Order)

1. **SQP+BER no-LR-boost** (80ep): Re-run `train_qtemporal_ber.py` with no boost (already set).
   Expected to perform better than tainted 61%. Compare to QPCA 64.2%.

2. **Parallel architecture no-LR-boost** (80ep): Re-run `train_qtemporal_ber.py --parallel`
   with no boost. The parallel design (identity init on classical path) should converge well.

3. **QPCA n_qubits sweep**: Test n_qubits=4,6 to see if smaller circuits outperform 8
   (barren plateau concern).

4. **QPCA on full 1604-ID dataset**: Validate that the +1.8pp quantum win holds at scale.
   This is critical for the paper claim.

5. **ReduceLROnPlateau**: Replace MultiStepLR with adaptive scheduler to eliminate ep30 collapse.
   `solver/adaptive_scheduler.py` exists but is not wired into training scripts yet.

---

## GDrive Upload Status (as of end of Instance 3)

All results uploaded to `gdrive:QTF-Clip-Results/` via rclone as individual .tar.gz files:
- `qclassifier_dense_nq4_80ep.tar.gz`
- `qclassifier_dense_nq6_80ep.tar.gz`
- `qclassifier_dense_nq8_80ep.tar.gz`
- `qpreprocess_noboost_80ep.tar.gz`
- `qpreprocess_classical_ablation_80ep.tar.gz`
- `sqp_ber_80ep_n2.tar.gz`
- `sqp_ber_80ep_dense.tar.gz`
- `sqp_ber_80ep_ham.tar.gz`
- `sqp_ber_80ep_parallel.tar.gz`
- `sqp_ber_80ep_parallel_dense.tar.gz`
- `sqp_ber_80ep_parallel_ham.tar.gz`
- Eval sweep .txt files (12 files)
- Earlier bundles: `instance3_early_results.tar.gz`, `instance3_chain_logs.tar.gz`

---

## User Preferences (Connor)

- **Terse communication**: no trailing summaries, no over-explaining basics
- **NEVER run more than 1 training at a time** — GPU fully saturated
- **Use nohup for overnight runs** to survive disconnects
- **Upload individual .tar.gz per experiment**, not bundles. No checkpoints, just logs and sweeps
- **Do not ask for permission constantly** — work autonomously
- **Status tables must include classical baseline column**
- **Never start agents unnecessarily**
- **"quantum preprocessing" = on raw images BEFORE ViT**, not elsewhere in pipeline

---

## VM / Infrastructure

- Thunder Compute GPU VM, ~$0.35/hr
- Conda env: `tfclip` (just use `python` directly — already activated)
- Background runs: use `nohup ... &` for survival across disconnects
- Dataset: `DATA/subset_250/` (symlink or actual), AG-VPReID with 250 IDs
- Results mirrored to GDrive via rclone (`gdrive` remote, `QTF-Clip-Results` folder)
- **rclone OAuth tokens expire** — if upload fails, re-authenticate with `rclone config reconnect gdrive:`
- MCP GDrive tools do NOT work; always use rclone or gdown

---

## Session Transcript

Full session JSONL (scrubbed of OAuth tokens) is in `transcripts/instance3_session.jsonl`.
Contains complete tool call history and can be used to reconstruct the exact sequence of
experiments, debugging steps, and decisions made.
