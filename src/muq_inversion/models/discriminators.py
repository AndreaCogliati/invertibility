"""Multi-period and multi-scale discriminators (CLAUDE.md Decoder Design item 6).

Added only after the reconstruction path passed Gate 3's smoke test with reconstruction-only
losses (mel + multi-res STFT) -- per CLAUDE.md's explicit ordering. Channel widths are kept
smaller than the original HiFi-GAN speech discriminators (1024-wide) to match this project's
"deliberately small" baseline; periods/scale count are the standard HiFi-GAN choices, which are
about waveform periodicity in samples (not speech-specific spectral content), so they're kept
as-is but exposed as config.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class PeriodDiscriminator(nn.Module):
    """Operates on a waveform reshaped into 2D frames of width `period`."""

    def __init__(self, period: int, channels: tuple[int, ...], kernel_size: int, stride: int):
        super().__init__()
        self.period = period
        in_channels = (1, *channels[:-1])
        self.convs = nn.ModuleList(
            weight_norm(
                nn.Conv2d(
                    c_in,
                    c_out,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=((kernel_size - 1) // 2, 0),
                )
            )
            for c_in, c_out in zip(in_channels, channels, strict=True)
        )
        self.output_conv = weight_norm(
            nn.Conv2d(channels[-1], 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """waveform: (batch, num_samples) -> (logits (batch, -1), feature_maps)"""
        batch, num_samples = waveform.shape
        if num_samples % self.period != 0:
            pad = self.period - (num_samples % self.period)
            waveform = nn.functional.pad(waveform, (0, pad), mode="reflect")
            num_samples += pad
        x = waveform.view(batch, 1, num_samples // self.period, self.period)

        feature_maps = []
        for conv in self.convs:
            x = self.activation(conv(x))
            feature_maps.append(x)
        x = self.output_conv(x)
        feature_maps.append(x)
        return torch.flatten(x, 1, -1), feature_maps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(
        self,
        periods: tuple[int, ...],
        channels: tuple[int, ...],
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.sub_discriminators = nn.ModuleList(
            PeriodDiscriminator(period, channels, kernel_size, stride) for period in periods
        )

    def forward(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        real_outputs, fake_outputs, real_fmaps, fake_fmaps = [], [], [], []
        for sub in self.sub_discriminators:
            r_out, r_fmap = sub(real)
            f_out, f_fmap = sub(fake)
            real_outputs.append(r_out)
            fake_outputs.append(f_out)
            real_fmaps.append(r_fmap)
            fake_fmaps.append(f_fmap)
        return real_outputs, fake_outputs, real_fmaps, fake_fmaps


class ScaleDiscriminator(nn.Module):
    """Operates directly on the (possibly downsampled) raw waveform with 1D convolutions."""

    def __init__(self, channels: tuple[int, ...]):
        super().__init__()
        in_channels = (1, *channels[:-1])
        kernel_sizes = (15, 41, 41, 41, 41, 5)
        strides = (1, 2, 2, 4, 4, 1)
        groups = (1, 4, 16, 16, 16, 1)
        for c_in, c_out, g in zip(in_channels, channels, groups[: len(channels)], strict=True):
            if c_in % g != 0 or c_out % g != 0:
                raise ValueError(
                    f"scale_channels={channels}: group count {g} must evenly divide both "
                    f"in_channels={c_in} and out_channels={c_out}"
                )
        self.convs = nn.ModuleList(
            weight_norm(
                nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, groups=g, padding=k // 2)
            )
            for c_in, c_out, k, s, g in zip(
                in_channels,
                channels,
                kernel_sizes[: len(channels)],
                strides[: len(channels)],
                groups[: len(channels)],
                strict=True,
            )
        )
        self.output_conv = weight_norm(nn.Conv1d(channels[-1], 1, kernel_size=3, stride=1, padding=1))
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """waveform: (batch, num_samples) -> (logits (batch, -1), feature_maps)"""
        x = waveform.unsqueeze(1)
        feature_maps = []
        for conv in self.convs:
            x = self.activation(conv(x))
            feature_maps.append(x)
        x = self.output_conv(x)
        feature_maps.append(x)
        return torch.flatten(x, 1, -1), feature_maps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_scales: int, channels: tuple[int, ...]):
        super().__init__()
        self.sub_discriminators = nn.ModuleList(ScaleDiscriminator(channels) for _ in range(num_scales))
        self.downsamplers = nn.ModuleList(
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2) for _ in range(num_scales - 1)
        )

    def forward(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        real_outputs, fake_outputs, real_fmaps, fake_fmaps = [], [], [], []
        for i, sub in enumerate(self.sub_discriminators):
            if i > 0:
                real = self.downsamplers[i - 1](real.unsqueeze(1)).squeeze(1)
                fake = self.downsamplers[i - 1](fake.unsqueeze(1)).squeeze(1)
            r_out, r_fmap = sub(real)
            f_out, f_fmap = sub(fake)
            real_outputs.append(r_out)
            fake_outputs.append(f_out)
            real_fmaps.append(r_fmap)
            fake_fmaps.append(f_fmap)
        return real_outputs, fake_outputs, real_fmaps, fake_fmaps
