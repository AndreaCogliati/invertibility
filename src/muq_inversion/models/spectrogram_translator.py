"""Spectrogram-to-DAC translator for the positive-control experiment (`tasks/TASK-01.md`).

`TemporalConvNet` (`models/temporal_conv.py`) is reused byte-for-byte unmodified: every prior
MuQ-to-DAC experiment could feed it directly because MuQ's channel count (1024) happens to equal
DAC's latent channel count (1024), and `TemporalConvNet` is a purely residual design (`x +
GELU(conv(x))`) that requires equal input/output width throughout. The spectrogram representation
does not share that coincidence (128 mel bins, this task's suggested starting point, vs. DAC's
1024), so a channel-matching step is unavoidable.

The minimal such step -- and the only structural difference from the MuQ pipeline permitted by
this task's "do not introduce new architectures" instruction -- is a single learned linear
projection from mel-bin count to `TemporalConvNet`'s channel width, applied per-frame before the
unchanged network. This is the same kind of component CLAUDE.md's own Decoder Design section
already specifies generically ("A learned input projection from MuQ channels to decoder
channels") for exactly this reason; no prior MuQ-to-DAC experiment needed to instantiate it only
because the MuQ/DAC channel counts already matched.
"""

from __future__ import annotations

import torch
from torch import nn

from muq_inversion.models.temporal_conv import TemporalConvNet


class SpectrogramToDacTranslator(nn.Module):
    def __init__(self, input_channels: int, dac_channels: int = 1024):
        super().__init__()
        self.input_projection = nn.Linear(input_channels, dac_channels)
        self.temporal_conv = TemporalConvNet(channels=dac_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, num_frames, input_channels) -> (batch, num_frames, dac_channels)."""
        return self.temporal_conv(self.input_projection(x))

    @property
    def receptive_field_frames(self) -> int:
        return self.temporal_conv.receptive_field_frames
