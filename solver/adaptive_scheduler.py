"""
LR scheduler wrappers matching the WarmupMultiStepLR interface
(get_lr() + step()) used by processor_clipreid_stage2.

Both wrappers share the same two-phase structure:
  1. Linear warmup: LR ramps from base_lr * warmup_factor → base_lr over warmup_epochs
  2. Scheduler phase (begins after warmup):
       AdaptiveLRWrapper  → ReduceLROnPlateau (drops only when loss plateaus)
       CosineRestartWrapper → CosineAnnealingWarmRestarts (periodic smooth resets)
"""
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts


class _WarmupBase:
    """Shared warmup logic."""
    def __init__(self, optimizer, warmup_epochs, warmup_factor):
        self._optimizer = optimizer
        self._base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self._warmup_epochs = warmup_epochs
        self._warmup_factor = warmup_factor
        self._epoch = 0
        self._set_warmup_lr(0)

    def _set_warmup_lr(self, epoch):
        alpha = self._warmup_factor + (1.0 - self._warmup_factor) * epoch / max(self._warmup_epochs, 1)
        for pg, base_lr in zip(self._optimizer.param_groups, self._base_lrs):
            pg['lr'] = base_lr * alpha

    def get_lr(self):
        return [pg['lr'] for pg in self._optimizer.param_groups]

    def set_metric(self, val):
        pass


class AdaptiveLRWrapper(_WarmupBase):
    def __init__(self, optimizer, warmup_epochs=10, warmup_factor=0.1,
                 patience=5, factor=0.5, min_lr=1e-7):
        super().__init__(optimizer, warmup_epochs, warmup_factor)
        self._scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=factor, min_lr=min_lr,
        )
        self._metric = None

    def set_metric(self, val):
        self._metric = val

    def step(self):
        self._epoch += 1
        if self._epoch <= self._warmup_epochs:
            self._set_warmup_lr(self._epoch)
        elif self._metric is not None:
            self._scheduler.step(self._metric)


class CosineRestartWrapper(_WarmupBase):
    """Warmup then CosineAnnealingWarmRestarts.

    T_0: epochs per restart cycle (counted from end of warmup)
    T_mult: cycle length multiplier (1=constant, 2=doubling)
    """
    def __init__(self, optimizer, warmup_epochs=10, warmup_factor=0.1,
                 T_0=10, T_mult=1, eta_min=1e-7):
        super().__init__(optimizer, warmup_epochs, warmup_factor)
        # Restore true base LRs before cosine init so it captures correct base_lrs,
        # then drop back to warmup start. Without this, cosine cycles around the
        # warmup-reduced LR (10x too low).
        for pg, base_lr in zip(optimizer.param_groups, self._base_lrs):
            pg['lr'] = base_lr
        self._cosine = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
        )
        self._set_warmup_lr(0)

    def step(self):
        self._epoch += 1
        if self._epoch <= self._warmup_epochs:
            self._set_warmup_lr(self._epoch)
        else:
            self._cosine.step()
