---
name: slim_checkpoint_revert
description: How to revert slim checkpoint change if needed — saves only trainable weights (~5MB vs 405MB)
type: project
---

# Slim Checkpoint Change (March 2026)

## What was changed
All `train_q*.py` scripts (adapter, channel, classifier, featext, frame, gated, gated_ccg, interlaced, temporal) had their final save block changed from full to slim.

**Old block** (full 405MB save):
```python
    from utils.iotools import save_checkpoint as _save_ckpt
    _save_ckpt(
        model.state_dict(),
        is_best=True,
        fpath=os.path.join(cfg.OUTPUT_DIR, 'last_model.pth.tar'),
    )
    logger.info(f"Final model saved to {cfg.OUTPUT_DIR}/last_model.pth.tar")
```

**New block** (slim ~5MB save):
```python
    from utils.iotools import save_slim_checkpoint as _save_slim
    _save_slim(model, fpath=os.path.join(cfg.OUTPUT_DIR, 'last_model.pth.tar'))
    logger.info(f"Final model (slim) saved to {cfg.OUTPUT_DIR}/last_model.pth.tar")
```

Also added `save_slim_checkpoint` to `utils/iotools.py` (saves only `requires_grad=True` params).

## To revert
Replace the "new block" with the "old block" in any/all of the above scripts.
Remove `save_slim_checkpoint` from `utils/iotools.py` if desired.

## Why this was done
WSL2 was crashing during the 400MB `torch.save` at end of training. Slim saves are ~5MB → no crash.
All eval scripts already use `strict=False` so slim checkpoints load correctly.

**Why:** Recurring WSL crash at checkpoint write (every single training run).
**How to apply:** Keep slim saves unless a full checkpoint is explicitly needed for transfer/reload.
