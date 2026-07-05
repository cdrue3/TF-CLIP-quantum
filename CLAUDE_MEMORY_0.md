# CLAUDE_MEMORY_0 — TF-CLIP-Quantum Project Context

This file is a handoff document from the first major VM session. It captures all decisions, results,
architecture findings, and workflow rules a fresh Claude instance needs to continue this project.

Full session transcript: `transcripts/79b4d34a-c90e-4745-b162-fd7dca926d0b.jsonl`

---

## Project Overview

**Goal**: Extend TF-CLIP (a video-based person re-ID system) with quantum computing components
across the entire ML pipeline (preprocessing → feature extraction → classification → optimisation → post-processing).

**User**: Connor, QUT grad student. Prefers terse communication, frequent status checks.

**Stack**:
- Base model: TF-CLIP (ViT-B/16 CLIP backbone, temporal aggregation for tracklet re-ID)
- Dataset: MARS (video person re-ID), stored at `/home/ubuntu/TF-CLIP-quantum/DATA/`
- Framework: PyTorch + PennyLane (quantum circuits)
- Python env: `~/tfc/bin/activate`

---

## Environment & Infrastructure

```bash
cd /home/ubuntu/TF-CLIP-quantum
source ~/tfc/bin/activate
```

**Compute**: GCP VM with GPU (intermittent driver issues — if CUDA unavailable, restart VM).
**Cost**: ~$0.35/hr. Always include cost estimate with ETAs.

**GDrive results upload** (MCP GDrive tools are broken — auth scope issue, never use them):
```bash
rclone copy <local_dir> gdrive:QTF-Clip-Results/
rclone ls gdrive:QTF-Clip-Results/   # verify upload
```

---

## Repository Structure (key additions)

```
TF-CLIP-quantum/
├── quantum_models/
│   ├── make_model_qtemporal.py          # Temporal QNN (angle encoding, VQC at temporal pool)
│   ├── make_model_qtemporal_dense.py    # Dense angle encoding (2 feat/qubit: RY+RZ)
│   ├── make_model_qamplitude.py         # Amplitude embedding (full 768-dim, no squash)
│   ├── make_model_qclassifier.py        # Quantum classifier (VQC replaces FC head)
│   ├── make_model_qclassifier_pca.py    # QPCA + VQC two-stage classifier
│   ├── amplitude/                       # AmplitudeEmbedding circuit helpers
│   ├── angle/                           # Angle/dense angle encoding circuits
│   ├── classification/                  # VQC classifier heads
│   ├── feature_extraction/              # QuNN feature extractors
│   ├── optimisation/                    # QuantumTripletLoss, etc.
│   ├── postprocessing/                  # SwapTest reranking, Dürr–Høyer
│   └── preprocessing/                  # QHED, QPCA preprocessing
├── train_qtemporal.py                   # Train QTemporal
├── train_qtemporal_dense.py             # Train Dense QTemporal
├── train_qamplitude.py                  # Train QAT (Quantum Amplitude Temporal)
├── train_qclassifier.py                 # Train QClassifier
├── train_qclassifier_pca.py             # Train QPCA Classifier
├── eval_qtemporal.py                    # Eval QTemporal
├── eval_qtemporal_dense.py              # Eval Dense QTemporal
├── eval_qamplitude.py                   # Eval QAT
├── eval_qclassifier.py                  # Eval QClassifier
├── configs/
│   ├── vit_clipreid_qtemporal.yml
│   ├── vit_clipreid_qtemporal_dense.yml
│   ├── vit_clipreid_qamplitude.yml
│   └── vit_clipreid_qclassifier.yml
├── transcripts/                         # Claude session JSONL transcripts
└── CLAUDE_MEMORY_0.md                   # This file
```

---

## Architecture Summary & Results

### Classical Baseline
- **TF-CLIP (classical)**: Rank-1 ≈ **82.2%** on MARS (80 epochs, full training)
- All quantum results below are measured against this.

### Implemented Architectures

| Name | Description | Best Rank-1 | Notes |
|------|-------------|-------------|-------|
| **QTemporal** | Angle-encoded VQC replaces temporal mean-pool. 8q, 2-layer. | ~56–58% | VQC in eval path → SLOW eval |
| **QTD (Dense)** | QTemporal with dense angle encoding (2 feat/qubit: RY+RZ). 8q. | ~57% | Same eval speed issue |
| **QAT** | AmplitudeEmbedding on full 768-dim CLIP features, 10q. | ~55% (ep80) | Slow eval (~20 min/checkpoint) |
| **QClassifier** | VQC replaces FC classification head only. CLIP backbone intact. | ~58% | Fast eval (~2 min/checkpoint) ✓ |
| **QPCA Classifier** | Two-stage: 10q QPCA → 8q VQC classifier. | ~2% (ep20) | BARREN PLATEAU — abandoned |
| **QChannel Preprocess** | Quantum channel ops on image before ViT. | Not run yet | In codebase |
| **SwapTest Reranking** | Swap test for post-hoc reranking of retrieved results. | -20pp vs baseline | Hurts performance |
| **Dürr–Høyer** | Quantum minimum search for reranking. | Implemented | Not benchmarked |
| **QuantumTripletLoss** | Quantum-circuit-based triplet loss. | In codebase | Not benchmarked |
| **Dense 4q** | Dense angle, 4 qubits. Qubit bottleneck experiment. | 56.9% ep40 | |
| **Dense 8q** | Dense angle, 8 qubits. Qubit bottleneck experiment. | 57.0% ep40 | |

### Qubit Bottleneck Experiment Result
**Theory**: more qubits → less information lost in 768→n_qubits compression → better Rank-1.
**Result**: DISPROVED. Dense 4q ≈ Dense 8q (56.9% vs 57.0%). The bottleneck is not qubit count.
**Likely cause**: overfitting. Training acc_id1 ≈ 83% but Rank-1 = 57% indicates the quantum
temporal layer adds noise rather than information. The VQC may be acting as a regulariser that
hurts discriminative power.

---

## Critical Bugs & Known Issues

### 1. `do_inference_rrs` return value discarded (training loop)
**File**: `processor/processor_clipreid_stage2.py`, line ~260
```python
do_inference_rrs(cfg, model, val_loader, num_query)  # return discarded
```
`best_performance` is never updated during training, so train logs show Rank-1=0.0% at all checkpoints.
**Workaround**: Always run standalone `eval_q*.py` scripts after training; never trust train log Rank-1.

### 2. Logger bug — Rank-1 invisible in train_log.txt
`do_inference_rrs` logs to `"transreid.test"` logger which has no file handler. INFO messages
are printed to console but not written to `train_log.txt`.
**Workaround**: Same as above — run standalone eval scripts.

### 3. Slim checkpoints missing ViT backbone
Checkpoints save only trainable (quantum) params (~5MB). ViT backbone weights not saved.
At eval time, the model loads the pretrained CLIP backbone from disk + the quantum params from checkpoint.
This is correct behaviour — ensure `pretrain_choice: 'ViT-B/16'` is set in eval config.

### 4. QAT attribute name bug (FIXED)
`make_model_qamplitude.py` line 463 originally used `self.tqa` but attribute is `self.qat`.
Fixed: changed to `self.qat(img_feature.view(B, T, -1))`.

---

## Eval Speed — CRITICAL RULE

**Slow models** (VQC in eval/inference path — runs per tracklet × T frames):
- QTemporal, QTD (Dense), QAT
- ~20–45 min per checkpoint at 8–10 qubits
- State vector scales as 2^n: 4q=16, 8q=256, 10q=1024

**Fast models** (VQC bypassed at eval, NECK_FEAT='before'):
- QClassifier, adapter models, QChannel Preprocess
- ~2 min per checkpoint

**NEVER recommend QTemporal/QAT when user asks for "fastest to evaluate."**
If user wants fast eval iteration, use QClassifier or adapter variants.

---

## Workflow Rules

1. **Always run eval sweep after training** — train logs don't capture Rank-1 reliably (logger bug).
2. **Never run training and eval in parallel** — user explicitly requires sequential: eval first, then train.
3. **Always include classical baseline column** in any results/status table.
4. **Time the first checkpoint** before estimating total eval sweep time — don't guess from qubit count alone.
5. **GDrive uploads**: use rclone only. MCP GDrive tools are permanently broken (auth scope).
6. **Checkpoints live in `logs/`** — gitignored. Results/logs to GDrive via rclone.
7. **Slim checkpoints** (~5MB) are the norm. Full checkpoint with backbone would be ~400MB.

---

## Pending Work (as of session end)

- [ ] **Root cause of quantum underfitting**: All quantum temporal models plateau ~56–58% vs 82% classical.
      Training acc_id1 is high (0.83+) but Rank-1 is low — overfitting/noise hypothesis.
- [ ] **QAT full eval sweep** — only ep80 was evaluated (55%). Run ep5–ep75 sweep.
- [ ] **Dense 12q** — was queued to complete the bottleneck experiment but likely unnecessary
      given 4q≈8q result.
- [ ] **QClassifier deeper run** — most promising fast-eval architecture, may benefit from longer training.
- [ ] **ReduceLROnPlateau** — MultiStepLR causes ep30 collapse in all quantum models. Switch to
      ReduceLROnPlateau for more stable training (fix identified, not implemented).
- [ ] **Quantum preprocessing** — QHED, QPCA-as-preprocessor (on raw images before ViT backbone)
      not yet benchmarked.

---

## GDrive Results Folder

All result logs and eval sweeps are uploaded to `gdrive:QTF-Clip-Results/`.
Use `rclone ls gdrive:QTF-Clip-Results/` to see what's there.

Key files there as of session end:
- `qgt_results.txt` — QTemporal (QGT) 20-epoch baseline
- `qclassifier_80ep_results.txt` — QClassifier 80ep eval sweep
- `qpca_classifier_results.txt` — QPCA barren plateau run (cancelled ep21)
- `qat_80ep_results.txt` — QAT ep80 eval
- `dense_4q_eval_results.txt` — Dense 4q sweep (ep10–ep80)
- `dense_8q_ep40_eval.txt` — Dense 8q ep40 only
- `ARCHITECTURE_KEY.txt` — Full key for all architecture names + qubit bottleneck summary

---

## How to Resume Training a Model

```bash
source ~/tfc/bin/activate
cd /home/ubuntu/TF-CLIP-quantum

# Example: resume QClassifier from ep80 checkpoint
python train_qclassifier.py \
    --config_file configs/vit_clipreid_qclassifier.yml \
    MODEL.PRETRAIN_CHOICE 'ViT-B/16' \
    OUTPUT_DIR logs/qclassifier_resume

# Example: eval sweep QClassifier checkpoints
for ep in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80; do
    ckpt="logs/qclassifier/transformer_${ep}.pth.tar"
    [ -f "$ckpt" ] || continue
    python eval_qclassifier.py \
        --config_file configs/vit_clipreid_qclassifier.yml \
        --checkpoint "$ckpt" \
        TEST.IMS_PER_BATCH 32
done
```

---

## Paper Pipeline Mapping

The thesis/paper maps quantum components to a 5-stage pipeline:

1. **Pre-processing** — QHED (quantum edge detection), QPCA on raw images, QPIE encoding
2. **Feature Extraction** — QuNNs, QSVM, quantum autoencoders, dressed quantum circuits (core pattern)
3. **Classification** — VQC classifier (impl), QSVM, quantum metric learning, data re-uploading (intractable at 96 blocks)
4. **Optimisation** — Quantum annealing, QAOA, Grover weight search, QuantumTripletLoss (impl)
5. **Post-processing** — MRF/Ising annealing, QGAN, SwapTest reranking (impl, -20pp), Dürr–Høyer (impl)

**Note on "quantum preprocessing"**: This specifically means on raw images BEFORE the ViT backbone,
not feature-level operations. QHED and QPCA-on-images are the relevant techniques.

**Data re-uploading (QClassifierReupload)**: 96-block circuit is intractable on CPU (~57s/batch).
Architecture needs a full rethink before it can be benchmarked.
