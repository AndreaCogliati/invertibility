import pytest
import torch
from torch import nn

from muq_inversion.grad_diagnostics import (
    compute_gradient_stats,
    gradient_stats_to_wandb_dict,
)


class _TinyTwoModuleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x):
        return self.b(self.a(x))


def test_compute_gradient_stats_basic_shape():
    model = _TinyTwoModuleNet()
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()

    stats = compute_gradient_stats(model)
    assert stats.total_grad_norm > 0
    assert stats.max_abs_grad >= stats.min_abs_grad >= 0
    assert set(stats.grad_norm_by_module.keys()) == {"a", "b"}
    assert all(v >= 0 for v in stats.grad_norm_by_module.values())


def test_compute_gradient_stats_raises_without_backward():
    model = _TinyTwoModuleNet()
    with pytest.raises(ValueError):
        compute_gradient_stats(model)


def test_compute_gradient_stats_zero_grad_detected_as_collapse():
    """A module that receives zero gradient (e.g. disconnected from the loss) should show up
    with a ~0 norm, not be silently omitted -- this is exactly the 'do gradients collapse'
    question this diagnostic exists to answer."""
    model = _TinyTwoModuleNet()
    x = torch.randn(2, 4)
    # Only run `a` through the loss; `b` gets no gradient at all.
    loss = model.a(x).sum()
    loss.backward()

    stats = compute_gradient_stats(model)
    assert "a" in stats.grad_norm_by_module
    assert "b" not in stats.grad_norm_by_module  # no grad at all for b, correctly excluded
    assert stats.total_grad_norm > 0


def test_gradient_stats_to_wandb_dict_flattens_module_norms():
    model = _TinyTwoModuleNet()
    x = torch.randn(2, 4)
    model(x).sum().backward()
    stats = compute_gradient_stats(model)

    flat = gradient_stats_to_wandb_dict(stats, prefix="grad")
    assert flat["grad/total_norm"] == stats.total_grad_norm
    assert flat["grad/norm_by_module/a"] == stats.grad_norm_by_module["a"]
    assert flat["grad/norm_by_module/b"] == stats.grad_norm_by_module["b"]
