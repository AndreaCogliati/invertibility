"""Frozen MuQ encoder wrapper.

Only the feature contract in CLAUDE.md is used:

    with torch.inference_mode():
        output = muq(waveform_24khz, output_hidden_states=False)
        features = output.last_hidden_state

Nothing here ever assumes a specific frame rate or channel count -- both are
measured from real model output and returned to the caller so downstream
code (doctor, model adapters) can assert against measured values instead of
hard-coded constants.
"""

from __future__ import annotations

import dataclasses

import torch
from muq import MuQ


@dataclasses.dataclass(frozen=True)
class MuQProbeResult:
    input_sample_rate: int
    input_num_samples: int
    input_duration_seconds: float
    feature_shape: tuple[int, ...]
    feature_channels: int
    num_frames: int
    frames_per_second: float
    samples_per_frame: float
    dtype: str
    device: str
    is_finite: bool


class FrozenMuQEncoder:
    """Loads MuQ, freezes it, and keeps it in eval mode for the lifetime of the process."""

    def __init__(self, model_id: str, device: torch.device):
        self.model_id = model_id
        self.device = device
        self.model = MuQ.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, waveform_24khz: torch.Tensor) -> torch.Tensor:
        """waveform_24khz: shape (batch, num_samples) at 24 kHz. Returns last_hidden_state."""
        output = self.model(waveform_24khz.to(self.device), output_hidden_states=False)
        return output.last_hidden_state

    @torch.inference_mode()
    def probe(self, waveform_24khz: torch.Tensor, sample_rate: int) -> MuQProbeResult:
        """Run one forward pass and measure the feature geometry (CLAUDE.md MuQ Feature Contract)."""
        assert waveform_24khz.ndim == 2, "expected (batch, num_samples)"
        features = self.encode(waveform_24khz)
        num_samples = waveform_24khz.shape[-1]
        duration = num_samples / sample_rate
        num_frames = features.shape[1]
        frames_per_second = num_frames / duration if duration > 0 else float("nan")
        samples_per_frame = num_samples / num_frames if num_frames > 0 else float("nan")

        return MuQProbeResult(
            input_sample_rate=sample_rate,
            input_num_samples=num_samples,
            input_duration_seconds=duration,
            feature_shape=tuple(features.shape),
            feature_channels=features.shape[-1],
            num_frames=num_frames,
            frames_per_second=frames_per_second,
            samples_per_frame=samples_per_frame,
            dtype=str(features.dtype),
            device=str(features.device),
            is_finite=bool(torch.isfinite(features).all().item()),
        )
