"""HiFi-GAN-style generator with a proper multi-receptive-field (MRF) fusion block
(CLAUDE.md Decoder Design).

Revised after listening feedback on Gate 3's output (ringing/buzzing persisting even after
adversarial training and loss rebalancing -- see notes/DECISIONS.md#gate3-generator-capacity).
The original generator ran each upsampling stage's output through only two small residual blocks
*in sequence* (kernel sizes 3 and 7, one dilation each) -- a thin receptive field, especially
right after a strided transpose-conv, which is exactly the situation prone to comb-filter/ringing
artifacts. This version follows the original HiFi-GAN's MRF design: several residual blocks with
different kernel sizes run *in parallel* on the same input and their outputs are averaged, each
block itself containing multiple dilated-conv pairs (not just one).

Only reconstruction-path components 1-5 are implemented here: input projection, temporal
conditioning, upsampling, residual blocks, waveform head. Multi-period/multi-scale discriminators
(component 6) live in `discriminators.py`.
"""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    """Multiple (dilated, dilation=1) conv pairs applied sequentially, each with its own
    residual connection -- one "kernel size" branch of the multi-receptive-field fusion."""

    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            nn.Conv1d(
                channels, channels, kernel_size, dilation=d, padding=(kernel_size - 1) * d // 2
            )
            for d in dilations
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)
            for _ in dilations
        )
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, time) -> (batch, channels, time)"""
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            residual = x
            x = conv1(self.activation(x))
            x = conv2(self.activation(x))
            x = x + residual
        return x


class MultiReceptiveFieldFusion(nn.Module):
    """Runs several ResBlocks (different kernel sizes) in parallel on the same input and
    averages their outputs -- the actual "multi-receptive-field" mechanism."""

    def __init__(
        self, channels: int, kernel_sizes: tuple[int, ...], dilation_sizes: tuple[tuple[int, ...], ...]
    ):
        super().__init__()
        if len(kernel_sizes) != len(dilation_sizes):
            raise ValueError("kernel_sizes and dilation_sizes must have equal length")
        self.blocks = nn.ModuleList(
            ResBlock(channels, k, d) for k, d in zip(kernel_sizes, dilation_sizes, strict=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = None
        for block in self.blocks:
            y = block(x)
            total = y if total is None else total + y
        return total / len(self.blocks)


class MiniGenerator(nn.Module):
    """Small MuQ-conditioned waveform generator with HiFi-GAN-style MRF residual fusion."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        upsample_rates: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.upsample_rates = upsample_rates
        self.total_upsample = 1
        for rate in upsample_rates:
            self.total_upsample *= rate

        self.input_projection = nn.Conv1d(input_channels, base_channels, kernel_size=7, padding=3)

        self.upsample_convs = nn.ModuleList()
        self.mrf_blocks = nn.ModuleList()
        channels = base_channels
        for rate in upsample_rates:
            out_channels = max(channels // 2, 1)
            self.upsample_convs.append(
                nn.ConvTranspose1d(
                    channels,
                    out_channels,
                    kernel_size=rate * 2,
                    stride=rate,
                    padding=rate // 2 + rate % 2,
                    output_padding=rate % 2,
                )
            )
            self.mrf_blocks.append(
                MultiReceptiveFieldFusion(out_channels, resblock_kernel_sizes, resblock_dilation_sizes)
            )
            channels = out_channels

        self.activation = nn.LeakyReLU(0.1)
        self.output_head = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, num_frames, input_channels) -> waveform (batch, num_samples)"""
        x = features.transpose(1, 2)  # (batch, input_channels, num_frames)
        x = self.input_projection(x)
        for upsample_conv, mrf in zip(self.upsample_convs, self.mrf_blocks, strict=True):
            x = upsample_conv(self.activation(x))
            x = mrf(x)
        waveform = self.output_head(x)
        return waveform.squeeze(1)
