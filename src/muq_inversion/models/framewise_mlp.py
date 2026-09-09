"""A deliberately minimal framewise MLP: no temporal modeling, no attention, no convolution.

Applies the same `Linear -> GELU -> Linear -> GELU -> Linear` stack independently to every
frame -- each output frame depends only on its own input frame, never on neighbors. Any
temporal alignment (e.g. an input/output frame-rate mismatch) must happen before this module,
not inside it.
"""

from __future__ import annotations

import torch
from torch import nn


class FramewiseMLP(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, num_frames, input_channels) -> (batch, num_frames, output_channels)."""
        return self.net(x)
