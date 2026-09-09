"""Frozen DAC (Descript Audio Codec) encoder/decoder wrapper.

Uses `transformers.DacModel` (already available in this project's pinned
`transformers==4.55.0` -- no new dependency needed; the standalone
`descript-audio-codec` PyPI package was tried first and rejected because its
`descript-audiotools` dependency pins `protobuf<3.20`, which conflicts with
this project's `tensorboard>=2.21.0` requirement).

Like `encoder.py`'s `FrozenMuQEncoder`, nothing here assumes a specific frame
rate or channel count -- both are measured from real model output. The
config's advertised `hop_length` field (512) does NOT match the empirically
measured hop (320, i.e. the product of `downsampling_ratios: [2,4,5,8]`) --
confirmed by direct probing, not by trusting the printed config, per
CLAUDE.md's "do not assume the frame rate" rule.

The "DAC latent" used throughout this experiment is `quantized_representation`
(shape `(batch, 1024, num_frames)`) -- the tensor `DacModel.decode()` actually
consumes, not the pre-quantizer `projected_latents` (256-dim) or the discrete
`audio_codes`. This is the natural choice for a "predict then decode" pipeline:
whatever the MLP predicts must be directly decodable by the frozen DAC decoder.

Recreated identically on this branch from `experiment/muq-to-dac-mlp`'s version
(branches don't share commits in this project's workflow); no changes.
"""

from __future__ import annotations

import dataclasses

import torch
from transformers import DacModel


@dataclasses.dataclass(frozen=True)
class DacProbeResult:
    input_sample_rate: int
    input_num_samples: int
    input_duration_seconds: float
    latent_shape: tuple[int, ...]
    latent_channels: int
    num_frames: int
    frames_per_second: float
    samples_per_frame: float
    dtype: str
    device: str
    is_finite: bool


class FrozenDacEncoder:
    """Loads DAC, freezes it, and keeps it in eval mode for the lifetime of the process."""

    def __init__(self, model_id: str, device: torch.device):
        self.model_id = model_id
        self.device = device
        self.model = DacModel.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, waveform_24khz: torch.Tensor) -> torch.Tensor:
        """waveform_24khz: (batch, num_samples) at 24 kHz. Returns quantized_representation
        (batch, 1024, num_frames) -- the continuous tensor `decode()` consumes."""
        assert waveform_24khz.ndim == 2, "expected (batch, num_samples)"
        output = self.model.encode(waveform_24khz.unsqueeze(1).to(self.device))
        return output.quantized_representation

    @torch.inference_mode()
    def decode(self, quantized_representation: torch.Tensor) -> torch.Tensor:
        """quantized_representation: (batch, 1024, num_frames) -> waveform (batch, num_samples)."""
        output = self.model.decode(quantized_representation=quantized_representation.to(self.device))
        return output.audio_values

    @torch.inference_mode()
    def probe(self, waveform_24khz: torch.Tensor, sample_rate: int) -> DacProbeResult:
        """Run one forward pass and measure the latent geometry (mirrors MuQ's feature contract)."""
        latent = self.encode(waveform_24khz)
        return measure_latent_geometry(latent, waveform_24khz.shape[-1], sample_rate)


def measure_latent_geometry(latent: torch.Tensor, num_samples: int, sample_rate: int) -> DacProbeResult:
    """latent: (batch, channels, num_frames) -- note channels-first, unlike MuQ's frames-first
    `last_hidden_state`. Split out from `probe` so callers needing the raw tensor (e.g. for
    building training targets) can measure geometry without a second forward pass."""
    duration = num_samples / sample_rate
    num_frames = latent.shape[-1]
    frames_per_second = num_frames / duration if duration > 0 else float("nan")
    samples_per_frame = num_samples / num_frames if num_frames > 0 else float("nan")

    return DacProbeResult(
        input_sample_rate=sample_rate,
        input_num_samples=num_samples,
        input_duration_seconds=duration,
        latent_shape=tuple(latent.shape),
        latent_channels=latent.shape[1],
        num_frames=num_frames,
        frames_per_second=frames_per_second,
        samples_per_frame=samples_per_frame,
        dtype=str(latent.dtype),
        device=str(latent.device),
        is_finite=bool(torch.isfinite(latent).all().item()),
    )
