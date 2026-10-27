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
    def encode_all_hidden_states(self, waveform_24khz: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """waveform_24khz: (batch, num_samples) at 24 kHz. Returns the full `hidden_states` tuple
        from a single forward pass (`output_hidden_states=True`) -- index 0 is the frontend/
        embedding output (before any transformer block), index `i` for `i>=1` is the output after
        the `i`-th transformer block. `hidden_states[-1] == last_hidden_state` (verified by direct
        comparison in `scripts/inspect_muq_hidden_states.py`, not assumed). Used only by
        `experiment/muq-layer-sweep`; `encode()` above remains the single-layer path used
        everywhere else in this project."""
        output = self.model(waveform_24khz.to(self.device), output_hidden_states=True)
        return output.hidden_states

    @torch.inference_mode()
    def encode_block_output(self, waveform_24khz: torch.Tensor, block: int) -> tuple[torch.Tensor, dict]:
        """waveform_24khz: (batch, num_samples) at 24 kHz -> (batch, T, C) output of conformer block
        `block` (1-based; TASK-22), plus the hook-based index check of `task22.hidden_state_with_check`
        (raises if hidden_states[block] is not exactly that block's output)."""
        from muq_inversion.task22 import hidden_state_with_check

        return hidden_state_with_check(lambda x: self.model(x, output_hidden_states=True).hidden_states,
                                       self.model.model.conformer.layers, waveform_24khz.to(self.device), block)

    @torch.inference_mode()
    def probe(self, waveform_24khz: torch.Tensor, sample_rate: int) -> MuQProbeResult:
        """Run one forward pass and measure the feature geometry (CLAUDE.md MuQ Feature Contract)."""
        assert waveform_24khz.ndim == 2, "expected (batch, num_samples)"
        features = self.encode(waveform_24khz)
        return measure_feature_geometry(features, waveform_24khz.shape[-1], sample_rate)


def measure_feature_geometry(
    features: torch.Tensor, num_samples: int, sample_rate: int
) -> MuQProbeResult:
    """Compute the MuQProbeResult stats for an already-computed `last_hidden_state` tensor.

    Split out from `FrozenMuQEncoder.probe` so callers that need the raw feature
    tensor (e.g. for caching) can measure geometry without a second forward pass.
    """
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
