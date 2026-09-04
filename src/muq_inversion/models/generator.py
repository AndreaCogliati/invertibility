"""Deliberately small HiFi-GAN-style generator (CLAUDE.md Decoder Design).

Only reconstruction-path components 1-5 are implemented: input projection,
temporal conditioning, upsampling, residual blocks, waveform head. Multi-
period/multi-scale discriminators (component 6) are deferred until the
reconstruction path passes its own smoke tests (Gate 3), per CLAUDE.md.

The upsample_rates product is a provisional placeholder for the Gate 0
shape-only forward pass. It must be replaced with the exact measured
MuQ samples-per-frame ratio (or a documented alignment/cropping strategy)
before any real training, per CLAUDE.md's MuQ Feature Contract and Decoder
Design sections.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """A single multi-receptive-field residual block: two dilated convs + skip."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, time) -> (batch, channels, time)"""
        residual = x
        x = self.conv1(self.activation(x))
        x = self.conv2(self.activation(x))
        return x + residual


class MiniGenerator(nn.Module):
    """Small MuQ-conditioned waveform generator for shape-validation smoke tests."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        upsample_rates: tuple[int, ...],
        residual_kernel_sizes: tuple[int, ...] = (3, 7),
        residual_dilations: tuple[int, ...] = (1, 3),
    ):
        super().__init__()
        self.upsample_rates = upsample_rates
        self.total_upsample = 1
        for rate in upsample_rates:
            self.total_upsample *= rate

        self.input_projection = nn.Conv1d(input_channels, base_channels, kernel_size=7, padding=3)

        self.upsample_stages = nn.ModuleList()
        channels = base_channels
        for rate in upsample_rates:
            out_channels = max(channels // 2, 1)
            stage = nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.ConvTranspose1d(
                    channels,
                    out_channels,
                    kernel_size=rate * 2,
                    stride=rate,
                    padding=rate // 2 + rate % 2,
                    output_padding=rate % 2,
                ),
                *(
                    ResidualBlock(out_channels, k, d)
                    for k, d in zip(residual_kernel_sizes, residual_dilations)
                ),
            )
            self.upsample_stages.append(stage)
            channels = out_channels

        self.output_head = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, num_frames, input_channels) -> waveform (batch, num_samples)"""
        x = features.transpose(1, 2)  # (batch, input_channels, num_frames)
        x = self.input_projection(x)
        for stage in self.upsample_stages:
            x = stage(x)
        waveform = self.output_head(x)
        return waveform.squeeze(1)
