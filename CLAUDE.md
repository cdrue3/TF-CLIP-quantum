# CLAUDE.md — Quantum TF-CLIP Project Context

This file exists so that a new Claude Code session (on any machine) can immediately understand this project without any prior conversation history. Read it fully before doing anything.

---

## Who is the user

Connor Claypool — undergraduate student at QUT, participating in the KIT IMPULSE Summer Research Program (April–July 2026). Supervisor at QUT. The research question is: **"Balancing classical and quantum contributions in a hybrid architecture for aerial-ground video person re-identification."** Target paper venue: WACV 2027.

Connor is technically capable and learns fast. He prefers terse, direct communication. Do not over-explain basics. Do not summarise what you just did — he can read the output.

---

## Project Overview

We are augmenting **TF-CLIP** (a classical video person re-ID model using CLIP + Vision Transformer) with **Variational Quantum Circuits (VQCs)** implemented in PennyLane. The goal is to find where, and under what conditions, a quantum component genuinely improves re-identification performance over the classical baseline.

The primary dataset is **AG-VPReID** — an aerial-ground video person re-ID dataset with both drone (aerial) and CCTV/wearable (ground) cameras. This is a hard, novel dataset that directly motivates quantum exploration because the aerial/ground viewpoint gap creates a challenging multi-modal retrieval problem.

---

## Environment

- **OS**: WSL2 on Windows
- **Conda env**: `tfclip` — just use `python` directly in the terminal (user is already in the env)
- **GPU**: single GPU, used by the CLIP backbone (ViT-B/16). VQCs run on CPU via PennyLane
- **Background runs**: must use `conda run -n tfclip python ...` — NOT bare `python`
- **Default**: give user the command to run themselves. Only background-run if they explicitly say "run it yourself" or "going AFK"

---

## Workflow Rules (IMPORTANT)

1. **Never run training in parallel with anything else.** GPU is fully saturated. At most two eval jobs can overlap (eval is lighter).
2. **AFK runs: one task at a time.** When user is AFK, launch each job as a separate background task. On completion notification, start the next. Never chain with `&&` for AFK runs — WSL2 inactivity crashes long silent chains.
3. **Never use `--fast_schedule` unless the user explicitly asks.** See below.
4. **All quantum train scripts save slim checkpoints** (~5MB). Eval scripts use `strict=False`. Do not revert this unless asked.

---

## Codebase Structure

```
TF-CLIP/
├── train.py                    # Classical TF-CLIP baseline training
├── train_q*.py                 # Quantum variant training scripts (one per architecture)
├── eval_agvpreid.py            # Eval Case 1 + Case 2 from a single checkpoint (~10 min total)
├── eval_*.py                   # Other eval scripts per architecture/dataset
├── configs/
│   ├── vit_clipreid_agvpreid.yml   # PRIMARY CONFIG — AG-VPReID, SEQ_LEN=8
│   ├── vit_clipreid_agreid.yml
│   └── vit_clipreid.yml            # Original TF-CLIP config (MARS/iLIDS, SEQ_LEN=4)
├── quantum_models/             # All quantum architectures
│   ├── quantum_temporal_agg.py         # QTemporal (TQA) — data re-uploading over T frames
│   ├── quantum_temporal_diff.py        # QTD — frame differences through VQC
│   ├── quantum_temporal_gated.py       # QGT — learned scalar gate on VQC correction
│   ├── quantum_frame_correlation.py    # QFC — all frame pairs through VQC
│   ├── quantum_gated_adapter.py        # QGated — gated adapter (best on AG-ReID)
│   ├── quantum_gated_adapter_ccg.py    # CCG — camera-conditioned gated adapter
│   ├── quantum_aerial_adapter.py       # ASQA — aerial-selective adapter (rejected)
│   └── make_model_*.py                 # Model builders for each architecture
├── datasets/
│   ├── set/agvpreid.py         # AG-VPReID dataset class
│   ├── set/agreid.py           # AG-ReID dataset class
│   └── make_dataloader_clipreid.py  # Uses rrs_test sampler, batch=32 for eval
├── processor/
│   └── processor_clipreid_stage2.py  # Training loop + CLIP memory disk cache
├── utils/iotools.py            # Contains save_slim_checkpoint
├── results.md                  # Comprehensive results across all datasets/architectures
├── temporal_plan.md            # Quantum temporal variant designs
├── architecture_explainer.md   # Detailed explanation of TF-CLIP + quantum components
└── DATA/
    └── AG-VPReID/
        ├── clip_memory_cache.pt    # Cached CLIP memory features — delete to regenerate
        └── scan_cache_*.json       # Dataset scan caches
```

---

## The TF-CLIP Model (What We're Augmenting)

TF-CLIP is a two-stage video person re-ID model:

**Stage 1** (CLIP Memory): Runs the CLIP text encoder over all training identity text prompts ("a photo of person X"). Builds cluster feature centroids per identity used in the I2T (image-to-text) contrastive loss in stage 2. Results cached to `DATA/AG-VPReID/clip_memory_cache.pt`.

**Stage 2** (Main Training): Fine-tunes the ViT backbone + trains classifier heads using:
- ID loss (cross-entropy over 1604 identity classes)
- Triplet loss (metric learning)
- I2T loss (image-to-text contrastive, weight=1.0)

**Temporal pooling**: Each tracklet has T=8 frames. Each frame → ViT → 768-dim vector. The T vectors are mean-pooled into a single tracklet descriptor. All quantum variants augment or replace this mean-pooling step.

**At eval time**: The 768-dim descriptor after the bottleneck layer (`NECK_FEAT='before'`) is used for retrieval. **Important**: When `NECK_FEAT='before'` (the default), adapter-style quantum modules are BYPASSED at eval — they only act as training regularizers. Only QTemporal/TQA runs at eval because it IS the pooling step.

---

## Quantum Architecture: How It Works

VQC structure (all architectures):
```
x [B, 768] → pre_net [Linear(768, n_qubits)] → AngleEmbedding → StronglyEntanglingLayers → measure → [2^n_qubits] → upscale [Linear(2^n_qubits, 768)] → residual on mean_pool
```

- **n_qubits=8, n_layers=2** for all AG-VPReID experiments
- **Device**: `default.qubit` with `diff_method='backprop'` — CPU only (no GPU acceleration)
- **Residual shortcut is critical**: QClassifier (no residual) always fails
- **Fewer qubits often better**: barren plateau kills gradients at more layers/qubits
- **PennyLane parameter broadcasting**: batch dimension passed directly to circuit (`[B, n_q]`) — ~3x speedup over per-sample loop

**GPU/CPU split**: ViT backbone runs on GPU. VQCs run on CPU. Data shuttles between them every batch. This is why VQC training is ~3-4x slower than classical (~0.5s/batch vs ~0.13s/batch). To fix this properly, would need TorchQuantum or lightning.gpu (cuQuantum) — both have compatibility issues with our PyTorch/CUDA setup.

---

## Dataset: AG-VPReID

- **Train**: 1604 identities, 13300 tracklets, 6 cameras (C0-C3 ground, C4-C5 aerial)
- **Eval Case 1**: aerial query → ground gallery (1459 IDs, 7320 query tracklets)
- **Eval Case 2**: ground query → aerial gallery (1459 IDs, 12775 gallery tracklets)
- **Case 1/Case 2 is eval-only** — training uses all tracklets from all cameras regardless
- **Data**: `DATA/AG-VPReID/` — not committed to git (too large)
- **SEQ_LEN=8** (official paper protocol — TF-CLIP original used SEQ_LEN=4)
- **Eval method**: `rrs_test` sampler, batch=32, ~10 min for both cases

**Important history**: We originally used `dense` eval (multi-clip averaging = test-time augmentation) which inflated numbers by ~2pp. Switched to `rrs_test` to match the official AG-VPReID paper protocol. The change is in `datasets/make_dataloader_clipreid.py`.

---

## Current Results (as of March 2026)

### AG-VPReID Classical Baseline (80ep, SEQ_LEN=4, rrs_test)
| Case | Rank-1 | Rank-5 | Checkpoint |
|------|--------|--------|------------|
| Case 1 (aerial→ground) | **65.1%** | 78.7% | `logs/agvpreid_classical_baseline_full/best_model.pth.tar` |
| Case 2 (ground→aerial) | **75.3%** | 85.8% | same |

This beats the paper's ~63% R1 because we train on the full 1604 IDs vs their 80/20 split.

### AG-VPReID 20ep SEQ_LEN=8 (Fair Comparison — all identical conditions)
| Model | ep8 acc_id1 | ep20 acc_id1 |
|---|---|---|
| Classical baseline | 0.073 | 0.275 |
| QTemporal VQC | 0.240 | 0.577 |
| QTD VQC | 0.246 | 0.571 |

VQC variants ~2-3x ahead of classical at same conditions. LR never decayed in 20ep so all models still learning — full Rank-1 eval pending (80ep runs needed).

### ASQA (80ep, SEQ_LEN=4) — REJECTED
Both VQC and classical ablation significantly underperform baseline. Aerial-selective masking disrupts the joint embedding space.

### AG-ReID Classical Baseline (80ep)
Rank-1: 74.3%, Rank-5: 86.9%
Best VQC: QGated adapter +4.6pp (78.9% vs 74.3%) — strongest VQC result overall.

Full results in `results.md`.

---

## Quantum Temporal Variants (Primary Current Focus)

### Implemented

**QTemporal / TQA** (`quantum_models/quantum_temporal_agg.py`, `train_qtemporal.py`)
Data re-uploading over all T=8 frames in a shared circuit. The VQC IS the temporal pooling — each frame's features are encoded sequentially, entangled, then the next frame re-encodes into the same qubits. Measurement produces the tracklet descriptor. Runs at eval time.

**QTD — Quantum Temporal Difference** (`quantum_models/quantum_temporal_diff.py`, `train_qtd.py`)
Computes T-1=7 consecutive frame differences, feeds them through VQC. Residual correction on mean_pool. Hypothesis: differences encode motion directly, lower variance than raw frames. Similar performance to QTemporal (0.571 vs 0.577 at 20ep).

### Implemented but not yet run at full scale

**QGT — Quantum-Gated Temporal** (`quantum_models/quantum_temporal_gated.py`, `train_qgt.py`)
Learned scalar gate `g = sigmoid(linear(mean_pool))` controls VQC contribution: `output = mean_pool + g * VQC_delta`. Gate initialised near 0.12. Logs gate values per camera to test Q2 (aerial vs ground gets different quantum contribution?). Run was killed before results due to SEQ_LEN=4 invalidity.

**QFC — Quantum Frame Correlation** (`quantum_models/quantum_frame_correlation.py`, `train_qfc.py`)
Processes all T(T-1)/2=28 frame pairs. Each pair [frame_i || frame_j] concatenated → 1536-dim → pre_net → VQC → average over all pairs → residual. Most novel but most expensive. Not yet run.

### Pending
- QGT 20ep run (SEQ_LEN=8)
- QFC 20ep run (SEQ_LEN=8)
- 80ep full runs for QTemporal + QTD (will need SEQ_LEN=8 baseline for fair comparison)
- SEQ_LEN=8 classical baseline 80ep run (for proper Rank-1 comparison)

---

## LR Schedule & fast_schedule

Default config steps: `[30, 50, 70]` with `GAMMA=0.1` — three 10x LR drops. For short runs (≤20ep) these never fire, so the model trains at peak LR throughout.

`--fast_schedule` flag (available on `train.py` and all `train_q*.py`): scales steps proportionally to 2 drops at [75%, 90%] of MAX_EPOCHS. **Never use unless explicitly requested by user.**

For 20ep comparison runs, it's intentional to NOT use fast_schedule — the model is still actively learning at ep20 and LR decay would hurt.

---

## Eval Protocol

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_agvpreid.py \
    --config_file configs/vit_clipreid_agvpreid.yml \
    --checkpoint <path/to/last_model.pth.tar> \
    INPUT.SEQ_LEN 8
```

- Always pass `INPUT.SEQ_LEN 8` to match training
- Eval uses `rrs_test` sampler (official protocol), batch=32
- ~10 min for both Case 1 + Case 2
- Results go to `/tmp/eval_*.txt` for logging

---

## Slim Checkpoints

All `train_q*.py` save only trainable weights (~5MB vs 405MB full checkpoint). This was done because WSL2 crashed on 400MB `torch.save`. Eval scripts use `strict=False` to load slim checkpoints correctly. `save_slim_checkpoint` is in `utils/iotools.py`.

---

## Key Architectural Findings

1. **Residual shortcut is critical** — no residual = VQC always loses to classical
2. **Fewer qubits better** — barren plateau kills gradients at >2 layers or >8 qubits
3. **Classical overshadowing** — large classical pre/post-processing layers dominate the VQC signal
4. **NECK_FEAT='before'** (default) means adapters are bypassed at eval — only TQA/QTemporal runs at inference
5. **QTemporal and QChannel** are the most promising architectures on AG-VPReID at 15ep
6. **Dataset size matters** — quantum advantage stronger on smaller datasets (AG-ReID 157 IDs > AG-VPReID 1604 IDs)
7. **ASQA rejected** — selectively applying VQC to aerial cameras disrupts the joint embedding space

---

## Research Questions

**Q1**: Which subtask/component benefits most from quantum computation?
→ Temporal aggregation (QTemporal) and channel attention (QChannel) are the candidates. Adapter-style modules seem to be overshadowed by classical layers.

**Q2**: Can input-adaptive routing (gated adapter) selectively apply quantum computation where it helps?
→ QGT directly addresses this. Camera-conditioned gating (CCG architecture) also explores it. The gate values per camera (aerial vs ground) are logged during training.

---

## Important Context on AG-ReID vs AG-VPReID

AG-ReID is a **video dataset** (it has tracklets — multiple frames per person) despite having only 157 train IDs and 2 cameras (ground + aerial). It is NOT image-only. However, it was superseded by AG-VPReID (the full large dataset) which has 1604 IDs and 6 cameras. All main results target AG-VPReID.

---

## Survey Paper Context

A survey paper "Toward Quantum-Enhanced Computer Vision" (Connor Druett et al., QUT/Monash/KIT) is at `Survey_Paper Revised draft.pdf`. Key findings relevant to this project:
- Multi-class degradation: VQC performance degrades sharply with class count (our 1604-class problem is hard)
- 768→8 dimensionality compression is a known bottleneck (sequential architecture flaw)
- Classical overshadowing is a documented problem
- Data re-uploading (what QTemporal uses) is validated as improving expressivity on NISQ devices
- Dense Angle encoding (2 features per qubit using phase) not yet tried — could double VQC information capacity

---

## Pending Work / Next Steps

1. **SEQ_LEN=8 classical baseline 80ep** — needed for fair Rank-1 comparison with QTemporal/QTD
2. **QGT 20ep** (SEQ_LEN=8) — gate analysis per camera
3. **QFC 20ep** (SEQ_LEN=8) — frame pair correlations
4. **QTemporal + QTD 80ep** (SEQ_LEN=8) — full Rank-1 eval
5. **GPU-accelerated VQC** — TorchQuantum is the most practical swap (circuits are native PyTorch ops)
6. **Dense Angle encoding** — try encoding 2 features/qubit to double VQC capacity

---

## Common Commands

```bash
# Classical baseline training (AG-VPReID, 80ep)
python train.py --config_file configs/vit_clipreid_agvpreid.yml \
    OUTPUT_DIR logs/agvpreid_classical_baseline_seq8

# QTemporal VQC training (20ep quick test)
python train_qtemporal.py --config_file configs/vit_clipreid_agvpreid.yml \
    --n_qubits 8 --n_layers 2 \
    SOLVER.STAGE2.MAX_EPOCHS 20 SOLVER.STAGE2.EVAL_PERIOD 999 \
    OUTPUT_DIR logs/agvpreid_qtemporal/vqc_seq8_20ep

# Eval
python eval_agvpreid.py --config_file configs/vit_clipreid_agvpreid.yml \
    --checkpoint logs/agvpreid_classical_baseline_full/best_model.pth.tar \
    INPUT.SEQ_LEN 8

# Delete CLIP memory cache (forces stage1 rebuild with new SEQ_LEN)
rm DATA/AG-VPReID/clip_memory_cache.pt
```
