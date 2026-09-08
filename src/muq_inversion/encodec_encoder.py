"""Frozen EnCodec encoder wrapper (encodec-hifigan-experiment branch only).

Not part of the MuQ baseline -- see notes/DECISIONS.md#encodec-hifigan for why this branch
exists (testing whether HiFi-GAN's reconstruction ceiling is set by the decoder family or by
MuQ's representation specifically, using EnCodec as an alternative representation source with
the exact same decoder/training recipe).

Uses the continuous pre-quantization encoder embedding (`EncodecModel.encoder(...)`), not the
quantized RVQ codes, to stay analogous to MuQ's continuous `last_hidden_state` -- both are
"the final time-resolved encoder representation" for their respective models, per this project's
existing feature-contract philosophy: never assume frame rate or channel count, measure them from
a real forward pass (mirrors CLAUDE.md's MuQ Feature Contract section, generalized to this
alternative encoder).
"""

from __future__ import annotations

import torch
from transformers import EncodecModel

from .encoder import MuQProbeResult, measure_feature_geometry


class FrozenEncodecEncoder:
    """Loads EnCodec, freezes it, and keeps it in eval mode for the lifetime of the process."""

    def __init__(self, model_id: str, device: torch.device):
        self.model_id = model_id
        self.device = device
        self.model = EncodecModel.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, waveform_24khz: torch.Tensor) -> torch.Tensor:
        """waveform_24khz: shape (batch, num_samples) at 24 kHz.

        Returns the continuous pre-quantization latent, shape (batch, num_frames, hidden_size)
        -- transposed from EncodecEncoder's native (batch, hidden_size, num_frames) conv-output
        layout to match the (batch, frames, channels) convention used throughout this project
        (matching MuQ's `last_hidden_state` layout).
        """
        x = waveform_24khz.unsqueeze(1).to(self.device)  # (batch, 1, num_samples)
        embeddings = self.model.encoder(x)  # (batch, hidden_size, num_frames)
        return embeddings.transpose(1, 2)

    @torch.inference_mode()
    def probe(self, waveform_24khz: torch.Tensor, sample_rate: int) -> MuQProbeResult:
        """Run one forward pass and measure the feature geometry (Gate-0-style diagnostics)."""
        assert waveform_24khz.ndim == 2, "expected (batch, num_samples)"
        features = self.encode(waveform_24khz)
        return measure_feature_geometry(features, waveform_24khz.shape[-1], sample_rate)
