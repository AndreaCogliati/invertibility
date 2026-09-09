"""Vendored from https://github.com/jik876/hifi-gan/blob/master/models.py (MIT License).

Near-verbatim: only mechanical changes (module-level constants instead of a `h` hyperparameter
namespace, explicit `__init__` arguments instead of an attribute bag, relative import of
`init_weights`/`get_padding`). No architectural change. `ResBlock2` and the spectral-norm-only
discriminator path are omitted -- the reference's own `config_v1.json` never selects them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from ._hifigan_utils import get_padding, init_weights

LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, int, int] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                )
                for _ in dilation
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class Generator(nn.Module):
    """Reference HiFi-GAN generator, `config_v1.json` defaults:
    upsample_rates=(8,8,2,2), upsample_kernel_sizes=(16,16,4,4), upsample_initial_channel=512,
    resblock_kernel_sizes=(3,7,11), resblock_dilation_sizes=((1,3,5),(1,3,5),(1,3,5)),
    num_mels=80 (input channels)."""

    def __init__(
        self,
        num_mels: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        upsample_initial_channel: int,
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, int, int], ...],
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, num_mels, num_frames) -> waveform (batch, 1, num_samples)"""
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                branch = self.resblocks[i * self.num_kernels + j](x)
                xs = branch if xs is None else xs + branch
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
