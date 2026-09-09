"""Vendored from https://github.com/jik876/hifi-gan/blob/master/utils.py (MIT License).

Only `init_weights`/`get_padding` -- the plotting/checkpoint-scanning helpers in the original
`utils.py` are irrelevant here (this project has its own checkpoint utilities).
"""

from __future__ import annotations

from torch import nn


def init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Reference HiFi-GAN's Conv weight init -- notably different from PyTorch's default Conv
    init (Kaiming uniform); applied via `.apply(init_weights)` on `nn.ModuleList`s of Conv
    layers, matching the original exactly."""
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        module.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)
