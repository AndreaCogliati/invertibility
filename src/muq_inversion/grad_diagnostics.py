"""Gradient-statistics diagnostics required by `tasks/TASK-03.md` ("Specifically monitor whether
gradients collapse or explode"). Pure function operating on any `nn.Module` right after
`loss.backward()` and before `optimizer.zero_grad()` -- unit-tested against a tiny synthetic
module, independent of this project's actual training scripts.
"""

from __future__ import annotations

import dataclasses

from torch import nn


@dataclasses.dataclass(frozen=True)
class GradientStats:
    total_grad_norm: float
    max_abs_grad: float
    min_abs_grad: float
    grad_norm_by_module: dict[str, float]


def compute_gradient_stats(model: nn.Module) -> GradientStats:
    """Call after `loss.backward()`, before `optimizer.step()`/`zero_grad()`.

    `grad_norm_by_module` is keyed by each of `model`'s direct named children (e.g.
    `input_projection`, `temporal_conv` for `SpectrogramToDacTranslator`) -- the L2 norm of that
    submodule's own gradients, pooled over all its parameters. Parameters with no gradient
    (`requires_grad=False`, or never touched by this backward pass) are skipped; raises if no
    parameter has a gradient at all, since that means `backward()` was not called or nothing in
    `model` is trainable -- a real bug this diagnostic should surface, not silently return NaNs.
    """
    all_grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not all_grads:
        raise ValueError("no gradients found -- was backward() called on a graph touching this model?")

    total_norm_sq = sum(float(g.norm(2) ** 2) for g in all_grads)
    max_abs = max(float(g.abs().max()) for g in all_grads)
    min_abs = min(float(g.abs().min()) for g in all_grads)

    grad_norm_by_module: dict[str, float] = {}
    for name, submodule in model.named_children():
        sub_grads = [p.grad.detach() for p in submodule.parameters() if p.grad is not None]
        if sub_grads:
            grad_norm_by_module[name] = float(sum(g.norm(2) ** 2 for g in sub_grads) ** 0.5)

    return GradientStats(
        total_grad_norm=total_norm_sq**0.5,
        max_abs_grad=max_abs,
        min_abs_grad=min_abs,
        grad_norm_by_module=grad_norm_by_module,
    )


def gradient_stats_to_wandb_dict(stats: GradientStats, prefix: str = "grad") -> dict[str, float]:
    """Flattens `GradientStats` into a flat dict suitable for `wandb.log(...)`."""
    out = {
        f"{prefix}/total_norm": stats.total_grad_norm,
        f"{prefix}/max_abs": stats.max_abs_grad,
        f"{prefix}/min_abs": stats.min_abs_grad,
    }
    for module_name, norm in stats.grad_norm_by_module.items():
        out[f"{prefix}/norm_by_module/{module_name}"] = norm
    return out
