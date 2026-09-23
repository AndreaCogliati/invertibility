"""The smallest reasonable temporal model: three stacked Conv1D(k=5) layers with residual
connections, replacing the framewise MLP's purely pointwise `Linear` layers with convolutions
that can see neighboring frames.

Operates on the *upsampled* (DAC-frame-rate) sequence, not the original 25 Hz MuQ sequence --
same temporal-alignment-by-repeat used by `experiment/muq-to-dac-mlp`'s framewise MLP (repeat
each MuQ frame to match DAC's exactly-3x-higher frame rate before this module). This is a
deliberate, single-variable change from that experiment: everything else (input, target,
optimizer, loss) stays identical, only the pointwise map is replaced by a local temporal one, to
isolate whether local temporal context -- not a wider or deeper network in general -- explains
any remaining gap.

Residual connections are trivial here because every layer keeps the same channel width (1024),
so `x + GELU(conv(x))` needs no projection.
"""

from __future__ import annotations

import torch
from torch import nn

KERNEL_SIZE = 5
NUM_LAYERS = 3


class TemporalConvNet(nn.Module):
    def __init__(self, channels: int = 1024, kernel_size: int = KERNEL_SIZE):
        super().__init__()
        padding = kernel_size // 2  # "same"-length convolution
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, num_frames, channels) -> (batch, num_frames, channels)."""
        h = x.transpose(1, 2)  # -> (batch, channels, num_frames), Conv1d's expected layout
        h = h + self.gelu(self.conv1(h))
        h = h + self.gelu(self.conv2(h))
        out = self.conv3(h)  # final projection: no activation/residual, unrestricted output range
        return out.transpose(1, 2)

    @property
    def receptive_field_frames(self) -> int:
        """Receptive field size, in frames of whatever sequence this module is applied to
        (the upsampled/DAC-frame-rate sequence, not the original MuQ sequence)."""
        return 1 + NUM_LAYERS * (self.conv1.kernel_size[0] - 1)
