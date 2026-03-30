---
name: AG-ReID Input-Adaptive Routing Plan
description: Why and what we're doing with camera-conditioned gating on AG-ReID
type: project
---

## Current Work: Camera-Conditioned Gating (CCG) on AG-ReID

**Why:** AG-ReID gated VQC was the only architecture where VQC beat classical (+4.6pp, 78.9% vs 74.3%). The hypothesis is that the aerial (cam 1, drone top-down) vs ground (cam 0, side-view) viewpoint gap creates fundamentally different CLIP feature distributions, and the gated adapter's learned scalar `g = sigmoid(gate_net(x))` may be routing quantum corrections differently per viewpoint.

**Critical finding:** ALL adapter models skip the adapter at eval when `TEST.NECK_FEAT='before'` (current config). So our AG-ReID results (78.9%, 77.0%, etc.) measure the adapter as a **training regularizer** on backbone features, NOT as an inference component. The gate never fires during retrieval. Exception: TQA always runs (it IS the pooling step).

**For input-adaptive routing to be testable, need `TEST.NECK_FEAT='after'` so the gate fires at eval.**

## Plan (saved in `/home/connor/.claude/plans/snug-gliding-nebula.md`)

**Phase 1:** Modify `eval_qgated.py` to:
1. Override `TEST.NECK_FEAT` to `'after'`
2. Add per-camera gate stratification (aerial vs ground gate distributions)
Then re-train gated VQC + classical with `NECK_FEAT='after'` for properly selected checkpoints.

**Phase 2:** Camera-Conditioned Gating (CCG) architecture:
- Add `cam_gate_embed = nn.Embedding(2, 16)` and widen `gate_net` to `Linear(768+16, 1)`
- Pass `cam_label` from model forward into the gate
- New files: `quantum_gated_adapter_ccg.py`, `make_model_gated_ccg.py`, `train_qgated_ccg.py`, `eval_qgated_ccg.py`

**How to apply:** When working on gated adapter experiments, always remember:
- NECK_FEAT='before' = adapter as training regularizer (adapter skipped at eval)
- NECK_FEAT='after' = adapter as inference component (adapter active at eval)
- All training commands for CCG/gating work must include `TEST.NECK_FEAT after`
