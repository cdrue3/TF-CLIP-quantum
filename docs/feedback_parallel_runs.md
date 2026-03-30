---
name: No parallel training runs
description: Never run training concurrently with anything else — GPU has no capacity
type: feedback
---

Never run a training job in parallel with anything else (other training or eval).

**Why:** GPU is fully saturated during training. Running eval or another training job alongside causes severe slowdown for all processes and risks OOM.

**How to apply:**
- Training always runs solo — wait for it to complete before starting anything else.
- The only acceptable parallelism is running two eval jobs simultaneously (eval is lighter and read-only).
- Queue is strictly sequential: train VQC → train classical → eval VQC + eval classical (those two can overlap) → next architecture's training.
