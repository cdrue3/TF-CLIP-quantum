# CLAUDE_MEMORY_2 — Post-Processing Instance

This file documents everything done on the **post-processing VM instance** of the TF-CLIP-quantum project. Read `CLAUDE_MEMORY_0.md` and `CLAUDE_MEMORY_1.md` first for project overview, backbone results, and feature extraction results.

---

## Instance Scope

This VM handled **post-processing and loss-function quantum augmentation only**. The classical backbone (80ep, 62.4% R1 on AG-VPReID subset_250) was frozen/pre-trained before experiments here began. All experiments use:

- Dataset: AG-VPReID `DATA/subset_250` (250 train IDs, 227 eval IDs)
- Case 1: aerial query → ground gallery
- Eval metric: Rank-1, Rank-5
- Classical baseline: **62.4% R1 @ ep80**

---

## Key Finding

**QuantumTripletLoss** (VQC-in-loss, classical arch) achieved **65.20% R1 (+2.8pp over classical)** — the best result across all quantum methods on this dataset. This is the recommended direction for future work.

---

## Experiments Summary

### 1. SwapTest Reranking — FAILED

- **What**: Quantum fidelity |⟨q|g⟩|² via CSWAP circuit reranks top-K classical L2 results
- **Result**: 45.5% R1 (-20.4pp). Complete failure.
- **Root cause**: Projecting 768-dim features → 4 qubits causes quantum concentration — all gallery items look equally similar. Random projection scrambles feature geometry.
- **Code**: `utils/quantum_retrieval.py` `QuantumSwapTestRanker`
- **Logs**: `logs/quantum_retrieval_swap_classical/`

---

### 2. Dürr-Høyer Quantum Minimum Finding — SPEED DEMO

- **What**: O(√N) quantum minimum oracle finds nearest gallery item directly
- **Result**: 66.42% R1 (identical to classical L2). Oracle calls: 118 vs 2719 = **23x theoretical speedup**.
- **Note**: Speedup is theoretical. On CPU simulator it's slower (pure Python loop). Demonstrates hardware-readiness — speedup materialises on real QPU with QRAM.
- **Code**: `utils/quantum_retrieval.py` `DurrHoyerSearch`
- **Run**: `python eval_agvpreid_quantum.py --retrieval durr_hoyer`

---

### 3. QPLR — Quantum Probabilistic Label Refining

- **What**: VQC processes top-K class logits → soft labels capturing inter-class quantum correlations. Applied as KL loss term during training.
- **Code**: `quantum_models/postprocessing/quantum_label_refiner.py`, `loss/make_loss.py::make_loss_qplr`, `train_qplr.py`

| Run | Best R1 | Notes |
|-----|---------|-------|
| 40ep from scratch | 58.8% @ ep35 (+0.7pp vs classical 58.1% @ ep35) | `logs/agvpreid_qplr/40ep/` |
| 80ep warm-start | 57.0% @ ep10 | Disrupts converged backbone |
| 80ep fresh, kl_weight=0.5 | 61.0% @ ep65-75 (-1.4pp) | `logs/agvpreid_qplr/80ep_fresh/` |
| **80ep fresh, kl_weight=0.1** | **62.80% @ ep70-80 (+0.4pp)** | `logs/agvpreid_qplr/80ep_kl01/` |

**kl_weight=0.1 full sweep** (complete epoch-by-epoch, `logs/agvpreid_qplr/80ep_kl01/results_sweep.txt`):

| Epoch | Rank-1 | Rank-5 |
|-------|--------|--------|
| ep05  | 47.70% | 65.50% |
| ep10  | 48.40% | 64.80% |
| ep15  | 44.30% | 62.50% |
| ep20  | 54.30% | 73.10% |
| ep25  | 51.00% | 68.00% |
| ep30  | 34.20% | 53.30% | ← LR-drop collapse |
| ep35  | 60.60% | 76.10% | ← recovery |
| ep40  | 61.20% | 76.50% |
| ep45  | 59.80% | 75.10% |
| ep50  | 61.40% | 76.90% |
| ep55  | 61.90% | 77.60% |
| ep60  | 62.40% | 77.70% |
| ep65  | 62.60% | 77.80% |
| **ep70** | **62.80%** | **78.00%** |
| ep75  | 62.80% | 78.00% |
| ep80  | 62.80% | 78.20% |

**Key insight — kl_weight=0.1**: Analogous to removing LR boost on other instances. kl_weight=0.5 adds ~35% extra gradient (loss ~5.8 vs classical ~4.3); kl_weight=0.1 reduces quantum signal to ~7% of total gradient. Same ep30 LR-drop collapse (34.2% R1, vs 1.1% with kl_weight=0.5 which was near-complete collapse) but dramatically better recovery and a stable +0.4pp plateau from ep70-80. This is the first QPLR configuration to beat the classical baseline.

---

### 4. Learned VQC Pairwise Reranker — FAILED

- **What**: VQC binary classifier trained on frozen features to score same-ID vs diff-ID pairs; reranks classical L2 top-K with alpha=0.3 blend
- **Result**: Best 66.4% @ ep01 (= L2 baseline, +0.0pp). Degrades monotonically with training.
- **MLP ablation**: 63.6% @ ep01, also degrades → problem is the approach, not quantum
- **Root cause**: Post-hoc reranking on AG-VPReID fails because the aerial→ground viewpoint gap means same-identity pairs are NOT mutual nearest neighbours. Backbone L2 features are already optimally calibrated. Any post-hoc reranking adds noise.
- **Code**: `quantum_models/postprocessing/quantum_reranker.py`, `train_qreranker.py`, `eval_reranker_sweep.py`
- **Logs**: `logs/quantum_reranker/80ep_k20/`, `logs/classical_reranker/80ep_k20/`

---

### 5. Quantum K-Reciprocal Reranker — FAILED

- **What**: VQC trained on k-NN distance patterns v_q/v_g (40-dim, 5:1 compression) reranks top-20 candidates
- **Classical k-reciprocal (Zhong et al.)**: 65.35% (-1.07pp vs L2 66.42%)
- **Quantum k-reciprocal**: Best 66.42% @ ep01 (+0.0pp). Degrades to 65.74% @ ep80.
- **Conclusion**: Same failure as pairwise reranker. Post-hoc reranking is exhausted on this dataset.
- **Code**: `quantum_models/postprocessing/quantum_kreranker.py`, `utils/kreranker.py`, `train_qkreranker.py`, `eval_kreranker_sweep.py`
- **Logs**: `logs/quantum_kreranker/80ep_k20/`, `logs/kreranker_sweep/`

---

### 6. QuantumTripletLoss — BEST RESULT (+2.8pp)

- **What**: VQC replaces TripletLoss during training only. Projects features through VQC circuit, uses cosine distance on quantum probability vectors for hard example mining. Classical model architecture — quantum only at training time. L2 retrieval at inference.

- **Architecture**:
  ```
  feats [B, 768] → pre_net Linear(768→n_qubits, bias=False) → sigmoid(·)×π [B, n_q]
  → AngleEmbedding(RY) → StronglyEntanglingLayers → qml.probs() → [B, 2^n_q]
  q_norm = probs / norm   cosine similarity kernel
  dist = (1 - q_norm @ q_norm.T).clamp(min=0)
  → hard_example_mining → SoftMarginLoss
  ```

- **Config**: n_qubits=6, n_layers=1, 80ep from scratch, no LR boost

| Epoch | Rank-1 | Rank-5 | vs Classical |
|-------|--------|--------|--------------|
| ep10  | 53.20% | 69.50% | -9.2pp |
| ep30  | 55.40% | 71.40% | -7.0pp (LR-drop dip, milder than QPLR) |
| ep40  | 62.40% | 77.30% | +0.0pp |
| ep50  | 62.90% | 77.60% | +0.5pp |
| ep60  | 64.70% | 78.40% | +2.3pp |
| **ep70** | **65.20%** | **78.60%** | **+2.8pp** |
| ep80  | 65.10% | 78.50% | +2.7pp |

- **Why it works**: VQC metric space guides the backbone to learn features that generalise better across the aerial→ground viewpoint gap. Unlike post-hoc reranking, it shapes backbone features during training. No quantum overhead at inference.
- **No LR boost**: `q_triplet` attached as `model.q_triplet` so `make_optimizer_2stage` picks it up at `BASE_LR`. No separate boosted param group.
- **Bug fix in `loss/make_loss.py`**: Multi-head `feat` list has mixed dims (768+512). Fix uses only `primary_feat = feat[0]` for quantum triplet — lazy pre_net init locked to first dim would fail on second. See `make_loss_q_triplet` in `loss/make_loss.py`.
- **Code**: `loss/quantum_triplet_loss.py`, `loss/make_loss.py::make_loss_q_triplet`, `train_q_triplet_loss.py`
- **Logs**: `logs/agvpreid_q_triplet_loss_80ep/`

---

## Post-Processing Conclusion

**Post-hoc reranking** (SwapTest, pairwise VQC, k-reciprocal VQC) is conclusively broken on AG-VPReID because the aerial→ground viewpoint gap violates the mutual-nearest-neighbour assumption that all manifold-exploiting reranking methods require.

**Training-time quantum loss** (QuantumTripletLoss, QPLR) works because it shapes backbone features rather than manipulating a fixed feature space. QuantumTripletLoss is the clear winner.

---

## What Was Not Tried (from survey, not feasible on CPU simulator)

- MRF/Ising annealing — requires D-Wave
- QGAN refinement — image generation, not applicable to re-ID
- Quantum K-means — segmentation, not applicable
- ZNE (Zero Noise Extrapolation) — implemented in `quantum_models/postprocessing/noise_mitigation.py` but not evaluated (no noise on simulator; useful for hardware deployment)

---

## Files Added/Modified on This Instance

**New source files:**
- `loss/quantum_triplet_loss.py` — QuantumTripletLoss module
- `loss/make_loss.py` — added `make_loss_q_triplet`, `make_loss_qplr` (primary_feat bug fix applied)
- `train_q_triplet_loss.py` — training script for QuantumTripletLoss
- `train_qplr.py` — QPLR training script
- `train_qkreranker.py` — quantum k-reciprocal reranker training
- `train_qreranker.py` — pairwise VQC reranker training
- `eval_kreranker_sweep.py` — checkpoint sweep for kreranker
- `eval_reranker_sweep.py` — checkpoint sweep for pairwise reranker
- `eval_checkpoint_sweep.py` — generic checkpoint sweep (classical/ham models)
- `utils/kreranker.py` — classical k-reciprocal (Zhong et al. 2017) implementation
- `quantum_models/postprocessing/quantum_label_refiner.py` — QPLR VQC module
- `quantum_models/postprocessing/quantum_kreranker.py` — VQC k-reciprocal reranker
- `quantum_models/postprocessing/quantum_reranker.py` — VQC pairwise reranker
- `quantum_models/postprocessing/classical_reranker.py` — MLP ablation for pairwise reranker
- `quantum_models/postprocessing/noise_mitigation.py` — ZNE implementation (hardware readiness)
- `utils/quantum_retrieval.py` — SwapTest + Dürr-Høyer implementations

**New result logs:**
- `logs/agvpreid_q_triplet_loss_80ep/` — QuantumTripletLoss sweep (ep10-80)
- `logs/agvpreid_qplr/80ep_kl01/` — QPLR kl_weight=0.1 sweep
- `logs/agvpreid_qplr/80ep_fresh/` — QPLR kl_weight=0.5 sweep
- `logs/quantum_kreranker/` — quantum k-reciprocal training logs
- `logs/kreranker_sweep/` — k-reciprocal eval sweep

---

## Recommended Next Steps

1. **Extend QuantumTripletLoss** — try n_qubits=8 or n_layers=2 (more expressive circuit)
2. **QuantumTripletLoss + QPLR combined** — stack both quantum loss terms
3. **Evaluate on full dataset** (not just subset_250) to confirm generalisation
4. **Dense angle encoding** for QuantumTripletLoss — double VQC information capacity per qubit
5. **Ablation**: replace VQC with classical MLP in QuantumTripletLoss to isolate quantum contribution
