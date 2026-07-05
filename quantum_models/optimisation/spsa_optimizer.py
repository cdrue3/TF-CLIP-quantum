"""
quantum_models/optimisation/spsa_optimizer.py

SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer for
quantum circuit parameters.

Survey finding: SPSA is more noise-robust than parameter-shift for deep
circuits (Friedrich & Maziero [21], Periyasamy et al. [66]). Cost is
independent of parameter count — requires exactly 2 forward passes per
step regardless of how many circuit weights there are.

This file provides:
  1. SPSAOptimizer — standard SPSA for a single param group
  2. make_hybrid_optimizer(model, cfg) — Adam for classical params,
     SPSA for all quantum circuit weights (identified by 'q_weights' in name)

Usage:
    optimizer_cls, optimizer_spsa = make_hybrid_optimizer(model, cfg)
    # In training loop:
    loss = loss_func(...)
    loss.backward()
    optimizer_cls.step(); optimizer_cls.zero_grad()
    # SPSA doesn't use .backward() gradients — calls forward twice internally
    optimizer_spsa.step(closure)  # closure = lambda: compute_loss_and_return()
    optimizer_spsa.zero_grad()
"""

import math
import torch
import torch.optim as optim


class SPSAOptimizer(optim.Optimizer):
    """
    Simultaneous Perturbation Stochastic Approximation optimizer.

    Estimates gradient by evaluating the objective at two points:
        θ+ = θ + c·Δ    (all params perturbed by +c in random direction)
        θ- = θ - c·Δ    (all params perturbed by -c)
    Gradient estimate: g_k = (f(θ+) - f(θ-)) / (2c) * Δ^{-1}

    With Rademacher perturbation (Δ_i ∈ {+1,-1}): gradient estimate is
    unbiased and independent of parameter count.

    Args:
        params  : iterable of parameters to optimize
        lr      : learning rate α (default 0.01)
        c       : perturbation magnitude (default 0.01)
        alpha   : lr decay exponent (default 0.602, standard SPSA)
        gamma   : c decay exponent (default 0.101, standard SPSA)
        A       : stability constant for lr decay (default 10)
    """

    def __init__(self, params, lr=0.01, c=0.01, alpha=0.602, gamma=0.101, A=10):
        defaults = dict(lr=lr, c=c, alpha=alpha, gamma=gamma, A=A)
        super().__init__(params, defaults)
        self._k = 0  # step counter

    @torch.no_grad()
    def step(self, closure):
        """
        Perform a single SPSA step.

        Args:
            closure: callable that re-evaluates the model and returns the loss.
                     Must call model.zero_grad() internally if needed.
        """
        self._k += 1
        k = self._k

        for group in self.param_groups:
            lr   = group['lr']
            c    = group['c']
            alpha = group['alpha']
            gamma = group['gamma']
            A     = group['A']

            # Decayed step sizes
            a_k = lr / (k + A) ** alpha
            c_k = c / k ** gamma

            # Collect all params in group as flat vectors for perturbation
            params = [p for p in group['params'] if p.requires_grad]
            if not params:
                continue

            # Rademacher perturbation: Δ_i ∈ {+1, -1} with equal probability
            deltas = [torch.randint_like(p, 0, 2).float() * 2 - 1 for p in params]

            # Evaluate at θ+ = θ + c_k·Δ
            for p, d in zip(params, deltas):
                p.add_(d, alpha=c_k)
            with torch.enable_grad():
                loss_plus = closure()

            # Evaluate at θ- = θ - 2c_k·Δ  (current state is θ+, so subtract 2c_k·Δ)
            for p, d in zip(params, deltas):
                p.add_(d, alpha=-2.0 * c_k)
            with torch.enable_grad():
                loss_minus = closure()

            # Restore to θ = θ- + c_k·Δ
            for p, d in zip(params, deltas):
                p.add_(d, alpha=c_k)

            # SPSA gradient estimate and update
            grad_est = (loss_plus - loss_minus) / (2.0 * c_k)
            for p, d in zip(params, deltas):
                # g_k_i = grad_est / Δ_i ; since Δ_i ∈ {±1}, 1/Δ_i = Δ_i
                p.add_(d, alpha=-a_k * grad_est.item())

        return loss_plus  # return last evaluated loss


def make_hybrid_optimizer(model, cfg):
    """
    Build a hybrid optimizer: Adam for classical params, SPSA for VQC weights.

    Quantum circuit weights are identified by containing 'q_weights' or
    'qlayer.weights' in their parameter name. Everything else uses Adam.

    Args:
        model: nn.Module with mixed classical and quantum parameters
        cfg:   YACS config (uses SOLVER.STAGE2.BASE_LR)

    Returns:
        (optimizer_adam, optimizer_spsa)
        Both should be stepped each batch; see module docstring for usage.
    """
    base_lr = cfg.SOLVER.STAGE2.BASE_LR

    classical_params = []
    quantum_params   = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'q_weights' in name or 'qlayer' in name:
            quantum_params.append(param)
        else:
            classical_params.append(param)

    weight_decay = getattr(cfg.SOLVER.STAGE2, 'WEIGHT_DECAY', 1e-4)
    optimizer_adam = torch.optim.Adam(
        classical_params, lr=base_lr, weight_decay=weight_decay
    )
    optimizer_spsa = SPSAOptimizer(
        quantum_params, lr=base_lr * 10, c=0.01
    )

    n_cls = sum(p.numel() for p in classical_params)
    n_q   = sum(p.numel() for p in quantum_params)
    print(
        f"[HybridOptimizer] Adam: {n_cls} classical params (lr={base_lr:.2e}); "
        f"SPSA: {n_q} quantum params (lr={base_lr*10:.2e})"
    )

    return optimizer_adam, optimizer_spsa
