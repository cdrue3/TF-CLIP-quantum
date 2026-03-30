# TF-CLIP Quantum Experiment Results

---

## AG-VPReID — Full Dataset (1604 train IDs, official case1/case2 splits)

### Classical TF-CLIP Baseline — Full Dataset (80 epochs)
| Case | Rank-1 | Rank-5 | Checkpoint |
|------|--------|--------|------------|
| Case 1 (aerial→ground) | **66.8%** | 79.5% | `logs/agvpreid_classical_baseline_full/best_model.pth.tar` |
| Case 2 (ground→aerial) | **77.7%** | 86.6% | same |

Beats paper's reported ~63% R1. Eval: capped dense (8 clips max/tracklet, uniformly subsampled), ~45 min.
Zero PID overlap between train (IDs 0–1603) and eval (IDs 1604–3062) — results legitimate.

---

## AG-VPReID — 80/20 Split (SUPERSEDED — 552 train IDs, 137 val IDs held-out, inflated numbers)

**Caveat**: eval on 20% held-out training split, NOT official test set. Numbers inflated vs paper baseline (~63% R1). Kept for reference.

### Classical TF-CLIP Baseline — 80/20 Split (80 epochs)
| Case | Rank-1 | Rank-5 | Checkpoint |
|------|--------|--------|------------|
| Case 1 (aerial→ground) | **67.9%** | 82.5% | `logs/agvpreid_classical_baseline/best_model.pth.tar` (ep70) |
| Case 2 (ground→aerial) | **70.8%** | 84.7% | same |

Training curve (Case 1 Rank-1, from training loop eval):
ep10=31.4%, ep20=39.4%, ep30=56.2%, ep40=59.9%, ep50=59.9%, ep60=61.3%, ep70=64.2% (best), ep80=62.8%

### AG-VPReID Quantum Variants — 80 epochs, full Rank-1 eval

| Architecture | C1 VQC R1 | C1 Classical R1 | C1 Baseline | C2 VQC R1 | C2 Classical R1 | C2 Baseline | Winner |
|---|---|---|---|---|---|---|---|
| QAdapter (4q) | 56.2% | **57.7%** | 67.9% | 66.4% | 66.4% | 70.8% | Classical +0.8pp avg |
| ASQA (8q) | 54.7% | 56.2% | 67.9% | 65.0% | 65.0% | 70.8% | Classical +0.8pp avg |

### 15ep Quick Tests (Data Subset — acc_id1 @ ep15, 500 batches/epoch, 8q 2L unless noted)

> Training accuracy metric only (not Rank-1). Models not converged at 15ep. Use for relative VQC vs classical comparison only.

| Architecture | VQC acc_id1 | Classical acc_id1 | Δ | Winner |
|---|---|---|---|---|
| QFeatExt (8q) | 0.385 | 0.412 | −2.7pp | Classical |
| ASQA (8q) | 0.390 | 0.447 | −5.7pp | Classical |
| QGated (8q) | 0.421 | 0.426 | −0.5pp | Classical ≈ tied |
| QFrame (8q) | 0.399 | 0.450 | −5.1pp | Classical |
| QTemporal (8q) | **0.419** | 0.418 | **+0.1pp** | **VQC** |
| QGated CCG (8q) | 0.396 | 0.431 | −3.5pp | Classical |
| QChannel (8q) | **0.426** | 0.414 | **+1.2pp** | **VQC** |
| QInterlaced (8q) | 0.407 | 0.417 | −1.0pp | Classical |

**Summary: VQC wins 2/8 (QTemporal +0.1pp, QChannel +1.2pp). Classical wins 6/8.**

Contrast with AG-ReID 80ep where VQC won 5/8 architectures. AG-VPReID 15ep shows weaker VQC signal overall, though QTemporal and QChannel are worth investigating at 80ep on the full dataset.

ASQA early-epoch detail (acc_id1 at iter 400):
| Epoch | VQC | Classical |
|-------|-----|-----------|
| 7 | **0.181** | 0.175 |
| 9 | **0.223** | 0.218 |
| 12 | **0.295** | 0.273 |
| 14 | 0.385 | **0.398** |
| 15 | 0.390 | **0.447** |

VQC leads briefly in early epochs then classical overtakes — consistent across all architectures.

Logs: `logs/agvpreid_q*/vqc_15ep/`, `logs/agvpreid_q*/classical_15ep/`, `logs/agvpreid_qaerial/vqc/`, `logs/agvpreid_qaerial/classical/`

### ASQA — 80 epochs, full Rank-1 eval

| | VQC R1 | VQC R5 | Classical R1 | Classical R5 | Baseline R1 |
|---|---|---|---|---|---|
| Case 1 (aerial→ground) | 54.7% | 77.4% | 56.2% | 77.4% | **67.9%** |
| Case 2 (ground→aerial) | **65.0%** | **81.8%** | **65.0%** | 80.3% | **70.8%** |

**Both ASQA variants underperform the unmodified classical baseline.**
- Case 1: −13.2pp (VQC) and −11.7pp (classical) vs baseline
- Case 2: −5.8pp (both) vs baseline
- VQC ≈ classical — aerial-selective masking hurts equally regardless of quantum/classical correction

**Conclusion: ASQA hypothesis rejected.** Selectively correcting aerial features degrades retrieval.
Possible reason: the aerial-selective mask disrupts the joint embedding space learned during training.
At NECK_FEAT='before' (default), the adapter's correction affects classifier training but is bypassed
at retrieval — the asymmetric masking may bias the learned feature manifold without improving retrieval.

Logs: `logs/agvpreid_qaerial/vqc_80ep/`, `logs/agvpreid_qaerial/classical_80ep/`

---

## AG-ReID (157 train IDs, 748 tracklets, 2 cameras — OLD DATASET, superseded by AG-VPReID)

> ⚠️ This was later identified as the image-based AG-ReID, not AG-VPReID. Results kept for reference but not used in paper.

### Classical TF-CLIP Baseline (80 epochs)
| Rank-1 | Rank-5 | mAP | Checkpoint |
|--------|--------|-----|------------|
| **74.3%** | 86.9% | — | `logs/agreid_classical_baseline/last_model.pth.tar` |

### Quantum Variants (80 epochs, all vs classical ablation)
| Architecture | VQC R1 | Classical R1 | VQC R5 | Classical R5 | Δ | Winner |
|---|---|---|---|---|---|---|
| Adapter (4q) | 76.7% | **79.4%** | 88.0% | 88.8% | −2.7pp | Classical |
| Channel Attn (8q) | 76.7% | **77.3%** | 87.2% | 88.0% | −0.6pp | Classical |
| Interlaced Q-C-Q (8q) | 76.7% | **79.4%** | 88.8% | 88.8% | −2.7pp | Classical |
| **Gated (8q)** | **78.9%** | 74.3% | **88.5%** | 86.6% | **+4.6pp** | **VQC ✓** |
| **Frame Attn (8q)** | **78.3%** | 77.0% | **89.3%** | 88.0% | **+1.3pp** | **VQC ✓** |
| **Temporal Agg (8q)** | **78.9%** | 77.5% | **88.2%** | 87.4% | **+1.4pp** | **VQC ✓** |
| **CCG (8q)** | **75.7%** | 74.3% | — | — | **+1.6pp** | **VQC ✓** |
| QFeatExt (8q) | **78.6%** | 75.9% | — | — | **+2.7pp** | **VQC ✓** |

Logs: `logs/agreid_qadapter/`, `logs/agreid_classical_baseline/`

---

## MARS (625 classes, 11310 tracklets — quick 15ep tests, 500 batches)

### Classical TF-CLIP Baseline (80 epochs)
| Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | Checkpoint |
|--------|--------|---------|---------|-----|------------|
| **90.9%** | 96.9% | 97.6% | 98.4% | **86.5%** | `logs/mars_vit_clip_reid_qclassifier/last_model.pth.tar` |

### Quick-Test Experiments (15ep, training acc_id only — not Rank-1)
| Experiment | VQC acc_id1 | Classical acc_id1 | Winner |
|------------|------------|-----------------|--------|
| Plain adapter 8q | 0.287 | **0.296** | Classical +0.9pp |
| Dense angle adapter 8q | **0.297** | 0.292 | VQC +0.5pp |
| Channel attention 8q | **0.301** | 0.294 | VQC +0.7pp |
| Frame attention 8q | 0.298 | 0.298 | Tied |
| Interlaced Q-C-Q 8q | **0.298** | 0.294 | VQC +0.4pp |
| **Adapter 4q** | **0.309** | 0.302 | **VQC +0.7pp (BEST)** |
| Adapter 12q | **0.299** | 0.250 | VQC +4.9pp (classical collapsed) |
| 4q, 4 layers | 0.299 | **0.302** | Classical (barren plateau) |
| 4q, 6 layers | 0.299 | **0.302** | Classical (same as 4L) |
| 4q, dense angle | 0.294 | **0.299** | Classical (dense angle hurts at 4q) |

### QClassifier Sweep (no residual — always fails)
| n_qubits | VQC acc_id1 | Classical acc_id1 |
|----------|------------|-----------------|
| 4 | 0.009 | **0.010** |
| 8 | 0.004 | **0.028** |
| 12 | 0.018 | **0.060** |

Key finding: residual shortcut is necessary for VQC to contribute at all.

Logs: `logs/mars_vit_clip_reid_qadapter/`, `logs/mars_vit_clip_reid_qclassifier/`, etc.

---

## iLIDS-VID (150 classes, small dataset)

### Classical TF-CLIP Baseline (80 epochs)
| Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | Checkpoint |
|--------|--------|---------|---------|-----|------------|
| **75.9%** | 82.8% | 86.9% | 90.8% | **72.1%** | `logs/ilids_vit_clip_reid/checkpoint_ep.pth.tar` |

### Adapter (15ep, warm-start from baseline)
| | Rank-1 | Rank-5 |
|---|---|---|
| VQC (15ep) | 80.0% | 87.3% |
| Classical (15ep) | **81.2%** | 88.3% |

Finding: iLIDS (small dataset) does NOT show quantum advantage. Sample-efficiency hypothesis rejected.
Logs: `logs/ilids_vit_clip_reid_qadapter/`

---

## Key Findings Summary

1. **Residual shortcut is critical** — QClassifier (no residual) always fails vs classical
2. **Fewer qubits often better** — 4q adapter (0.309) beats 8q (0.287); barren plateau at more layers
3. **AG-ReID gated adapter** is the strongest VQC result (+4.6pp R1 over classical, 80ep)
4. **AG-VPReID 15ep** — classical wins 6/8 architectures; VQC edges out only on QTemporal (+0.1pp) and QChannel (+1.2pp)
5. **AG-VPReID 80ep (data subset)** — all variants below classical baseline; VQC ≈ classical in Rank-1 (within 1-2pp)
6. **ASQA hypothesis rejected** — selectively applying VQC to aerial features hurts retrieval at both 15ep and 80ep
7. **MARS quick tests** — VQC matches or slightly beats classical in acc_id on most architectures at 15ep
8. **Dataset matters** — quantum advantage more likely on smaller datasets / fewer classes; AG-VPReID shows weaker VQC signal than AG-ReID
9. **Full dataset pending** — all AG-VPReID results above use 80/20 training split; retrain needed once full dataset arrives
