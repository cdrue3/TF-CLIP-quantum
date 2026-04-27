# TF-CLIP Project Memory

## IMPORTANT: Workflow Rules
**Default: give user the command to run themselves.** Only background-run when explicitly told ("run it yourself", "going AFK").
**Terminal env:** User has `tfclip` conda active — just use `python` directly in instructions.
**Exception:** `run_in_background=true` Bash calls must use `conda run -n tfclip python ...`.
**AFK runs: one task at a time.** Start next on completion notification. See `memory/feedback_afk_runs.md`.
**Never run training alongside anything.** At most two evals can run in parallel. See `memory/feedback_parallel_runs.md`.
- [feedback_fast_schedule.md](feedback_fast_schedule.md) — Never use --fast_schedule unless explicitly requested

## Slim Checkpoints
All `train_q*.py` final saves use `save_slim_checkpoint` (~5MB vs 405MB). Eval scripts use `strict=False`.
Revert details: `memory/feedback_slim_checkpoint.md`.

## Current Dataset: AG-VPReID (primary focus)
- **Full dataset**: 1604 train IDs, 13300 tracklets, 6 cameras. Aerial cams: C4, C5. Ground: C0-C3.
- Official case1/case2 query-gallery splits (no held-out hack). Config: `configs/vit_clipreid_agvpreid.yml`
- Data: `DATA/AG-VPReID/`, scan caches: `DATA/AG-VPReID/scan_cache_*.json`
- CLIP memory cached: `DATA/AG-VPReID/clip_memory_cache.pt` (delete to regenerate)
- Eval: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_agvpreid.py --checkpoint ...`
- Eval settings: rrs_test sampling, batch=32, num_workers=4 → ~5 min for both cases

## AG-VPReID Results

### Classical Baseline — Full Dataset (80 epochs, SEQ_LEN=4)
| Case | Rank-1 (rrs_test) | Rank-5 | Checkpoint |
|------|--------|--------|------------|
| Case 1 (aerial→ground) | **65.1%** | 78.7% | `logs/agvpreid_classical_baseline_full/best_model.pth.tar` |
| Case 2 (ground→aerial) | **75.3%** | 85.8% | same |

Eval method: rrs_test (official, batch=32, ~5 min). Dense eval gave 66.8%/77.7% (~2pp inflation).
Beats paper's ~63% R1 (paper used 80/20 split, we use full 1604 train IDs).
This is the new baseline to beat for all quantum variants.

### Quick Tests (15ep, acc_id1 only — training metric, no Rank-1)
| Architecture | VQC acc_id1 | Classical acc_id1 | Winner |
|---|---|---|---|
| QFeatExt 8q | 0.385 | 0.412 | Classical +2.7pp |
| ASQA 8q (aerial-selective) | 0.390 | **0.447** | Classical +5.7pp |
| QGated 8q | 0.421 | 0.426 | Classical +0.5pp |
| QFrame 8q | 0.399 | 0.450 | Classical +5.1pp |
| **QTemporal 8q** | **0.419** | 0.418 | **VQC +0.1pp** |
| QGated CCG 8q | 0.396 | 0.431 | Classical +3.5pp |
| **QChannel 8q** | **0.426** | 0.414 | **VQC +1.2pp** |
| QInterlaced 8q | 0.407 | 0.417 | Classical +1.0pp |

VQC wins 2/8: QTemporal (+0.1pp), QChannel (+1.2pp). Classical wins 6/8.
Logs: `logs/agvpreid_q*/vqc_15ep/`, `logs/agvpreid_q*/classical_15ep/`

### ASQA 80ep Rank-1 (full eval)
| Case | VQC R1 | Classical R1 | Baseline |
|---|---|---|---|
| Case 1 (aerial→ground) | 54.7% | 56.2% | **67.9%** |
| Case 2 (ground→aerial) | 65.0% | 65.0% | **70.8%** |
**Both underperform baseline. ASQA hypothesis rejected.** Selective masking disrupts joint embedding space.
Logs: `logs/agvpreid_qaerial/vqc_80ep/`, `logs/agvpreid_qaerial/classical_80ep/`

## AG-ReID Results (OLD dataset — 157 train IDs, 2 cameras, superseded)
Classical baseline: Rank-1 **74.3%**, Rank-5 86.9%

| Architecture | VQC R1 | Classical R1 | Winner |
|---|---|---|---|
| Adapter (4q) | 76.7% | **79.4%** | Classical |
| **Gated (8q)** | **78.9%** | 74.3% | **VQC +4.6pp** ✓ |
| **Frame Attn (8q)** | **78.3%** | 77.0% | **VQC +1.3pp** ✓ |
| **Temporal Agg (8q)** | **78.9%** | 77.5% | **VQC +1.4pp** ✓ |

Full table: `results.md`. Logs: `logs/agreid_*/`

## MARS Quick-Test Summary (15ep, 500 batches, 625 classes)
Classical baseline: Rank-1 90.9%, mAP 86.5% | `logs/mars_vit_clip_reid_qclassifier/last_model.pth.tar`

Best VQC config: **4q adapter, 2 layers, standard angle** (acc_id1=0.309 vs classical 0.302).
Key findings: residual is critical; fewer qubits better; more layers = barren plateau; qclassifier (no residual) always fails.
Full table + iLIDS results: `results.md`.

## Key Architectural Findings
1. **Residual shortcut critical** — QClassifier (no residual) always fails vs classical
2. **Classical ablation usually wins** at 15ep; VQC wins only QTemporal (+0.1pp) and QChannel (+1.2pp) on AG-VPReID subset
3. **Gated adapter (AG-ReID, 80ep): +4.6pp Rank-1** — strongest VQC result
4. **ASQA hypothesis rejected** — both 15ep and 80ep confirm selective aerial masking hurts retrieval
5. **Quantum kernel (IQP):** always hurts retrieval — quantum concentration kills discriminability
6. **Full dataset pending** — all AG-VPReID results used 80/20 training split; real test set arriving, retrain needed

## New Files (ASQA, March 2026)
- `quantum_models/quantum_aerial_adapter.py` — AerialSelectiveAdapter: hard mask `aerial_mask ∈ {C4,C5}`
- `quantum_models/make_model_aerial_adapter.py` — model builder (clone of CCG, swapped adapter)
- `train_qaerial.py` — training script (clone of train_qgated_ccg.py)

## Other Key Files
- `eval_agvpreid.py` — eval Case 1 + Case 2 from single checkpoint
- `datasets/set/agvpreid.py` — dataset class (80/20 split, case1/case2 query/gallery)
- `quantum_models/quantum_gated_adapter_ccg.py` — CCG adapter (cam-conditioned learned gate)
- `processor/processor_clipreid_stage2.py` — training loop + CLIP memory disk cache
- `results.md` — comprehensive results across all datasets/architectures

## Pending Architecture Ideas
- [idea_qft_parallel.md](idea_qft_parallel.md) — Parallel QFT + StronglyEntanglingLayers branch in VQC. QFT gives structured frequency mixing (no parameters, no barren plateau), SEL retains learnable expressivity. Novel framing for WACV paper. Revisit after Colab setup.

## KIT Program (April–July 2026)
Research: "Balancing classical and quantum contributions in a hybrid architecture". Paper: WACV 2027.
Research Q1: which subtask benefits from quantum? Q2: input-adaptive routing (gated adapter explores this).

## iLIDS-VID Baseline (80ep)
Rank-1 **75.9%**, Rank-5 82.8%, mAP 72.1% | `logs/ilids_vit_clip_reid/checkpoint_ep.pth.tar`
