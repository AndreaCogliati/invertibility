"""TASK-15 translator: MuQ last_hidden_state -> BigVGAN-v2 log-mel.

MuQ runs at 25 Hz (960-sample hop at 24 kHz); BigVGAN-v2's mel runs at 93.75 Hz (256-sample hop).
The ratio, 3.75, is not an integer, so the repeat-interleave alignment used for EnCodec (exactly
3x) does not apply. The simplest exact alternative is linear interpolation of the projected MuQ
sequence to the target mel length, then the project's existing `TemporalConvNet` at the mel frame
rate, then a per-frame projection to the mel bins:

    (B, T_muq, 1024) -> Linear(1024, hidden) -> linear interp to T_mel -> TemporalConvNet(hidden)
        -> Linear(hidden, num_mels) -> (B, num_mels, T_mel)

`hidden` is decoupled from `num_mels` (100) so the temporal block isn't forced to the output width.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from muq_inversion.models.temporal_conv import TemporalConvNet


class MuQToMelTranslator(nn.Module):
    def __init__(self, input_channels: int = 1024, hidden_channels: int = 256, num_mels: int = 100):
        super().__init__()
        self.input_projection = nn.Linear(input_channels, hidden_channels)
        self.temporal_conv = TemporalConvNet(channels=hidden_channels)
        self.output_projection = nn.Linear(hidden_channels, num_mels)

    def forward(self, muq_features: torch.Tensor, num_mel_frames: int) -> torch.Tensor:
        """muq_features: (B, T_muq, input_channels) -> log-mel (B, num_mels, num_mel_frames)."""
        h = self.input_projection(muq_features).transpose(1, 2)  # (B, hidden, T_muq)
        h = F.interpolate(h, size=num_mel_frames, mode="linear", align_corners=False)
        h = self.temporal_conv(h.transpose(1, 2))  # (B, T_mel, hidden)
        return self.output_projection(h).transpose(1, 2)

    @property
    def receptive_field_mel_frames(self) -> int:
        return self.temporal_conv.receptive_field_frames
