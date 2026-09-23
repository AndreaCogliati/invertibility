"""Frozen EnCodec encoder/decoder wrapper, for `tasks/TASK-10.md`'s DAC-vs-EnCodec shift-
sensitivity replication only -- no other task in this project trains against or conditions on
EnCodec.

Uses `transformers.EncodecModel` (`facebook/encodec_24khz`, the same checkpoint this project used
on its now-separate `encodec-hifigan-experiment` branch -- see `notes/EXPERIMENT_SYNTHESIS.md`),
already available via this project's pinned `transformers==4.55.0`, no new dependency.

## Measured geometry (probed once, not assumed -- CLAUDE.md's "do not assume the frame rate" rule)

- `sampling_rate`: 24000 Hz (matches this project's `SAMPLE_RATE` throughout).
- `frame_rate`: 75 Hz -- identical to DAC's own measured 75 Hz (`dac_encoder.py`), so latents
  align frame-for-frame with no repeat-interleave/resampling step needed for any DAC-vs-EnCodec
  comparison at matching shift magnitudes.
- `num_quantizers` (max, at the highest supported bandwidth, 24.0 kbps): 32 -- identical to DAC's
  32 codebooks, `codebook_size=1024` also identical. `DEFAULT_BANDWIDTH=24.0` (this module's
  choice) uses all 32 quantizers, mirroring DAC's own convention throughout this project of using
  full RVQ depth (`n_quantizers=None` in `DacResidualVectorQuantize.forward`) rather than a
  reduced bandwidth -- verified via `EncodecResidualVectorQuantizer.get_num_quantizers_for_bandwidth`.
- Continuous (pre-quantizer) latent dimensionality: 128 -- notably smaller than DAC's 1024-dim
  `quantized_representation`; the quantized (post-RVQ) representation shares this same 128-dim
  shape (unlike DAC, whose pre- and post-quantizer tensors are both 1024-dim already).
- Decoder output length: exactly matches input length (verified: 48000 samples in -> 48000 out),
  unlike DAC's own measured behavior (48000 in -> 47992 out, 8 samples shorter) -- a real,
  documented EnCodec-vs-DAC preprocessing/architecture difference relevant to any audio-domain
  alignment code reused from TASK-08.

## Continuous vs. quantized representation (`tasks/TASK-10.md`'s Representation A / B)

- Representation A (continuous): `self.model.encoder(input_values)` -- the pre-quantizer
  embedding, analogous to DAC's `pre_quantizer_encoding`.
- Representation B (quantized): `self.model.quantizer.decode(self.model.quantizer.encode(...))`
  -- the post-RVQ continuous representation `EncodecModel.decode()` actually consumes, the closest
  analog to DAC's `quantized_representation` and the representation `encode()`/`encode_stages()`
  below return as "the EnCodec latent" for every main analysis in TASK-10, matching how DAC's own
  `quantized_representation` was used as "the DAC latent" throughout TASK-07/08/09.

Verified once (this module's own test suite) that the manual `encoder -> quantizer.encode ->
quantizer.decode -> decoder` path reproduces `EncodecModel.encode()`/`.decode()`'s top-level
output exactly -- no monkeypatch of `transformers` internals needed, mirroring `dac_encoder.py`'s
own verification discipline.
"""

from __future__ import annotations

import dataclasses

import torch
from transformers import EncodecModel

DEFAULT_BANDWIDTH = 24.0  # kbps -- uses all 32 quantizers, mirroring DAC's full-RVQ-depth convention


@dataclasses.dataclass(frozen=True)
class EncodecEncodeStages:
    """Both stages of EnCodec's encode/decode pipeline, split apart -- mirrors `dac_encoder.py`'s
    `DacEncodeStages`. `audio_codes` here is already transposed to (batch, num_quantizers,
    num_frames), matching `DacEncodeStages.audio_codes`'s layout so
    `dac_shift_sensitivity_common.codebook_agreement` works unchanged on either model's codes."""

    pre_quantizer_encoding: torch.Tensor  # (batch, 128, num_frames), before RVQ
    quantized_representation: torch.Tensor  # (batch, 128, num_frames), after RVQ
    audio_codes: torch.Tensor  # (batch, num_quantizers, num_frames), per-stage codebook indices


class FrozenEncodecEncoder:
    """Loads EnCodec, freezes it, and keeps it in eval mode for the lifetime of the process --
    mirrors `dac_encoder.py`'s `FrozenDacEncoder` interface exactly (`encode`, `encode_stages`,
    `decode`) so `tasks/TASK-10.md`'s script can reuse the same call patterns as TASK-07/08/09."""

    def __init__(self, model_id: str, device: torch.device, bandwidth: float = DEFAULT_BANDWIDTH):
        self.model_id = model_id
        self.device = device
        self.bandwidth = bandwidth
        self.model = EncodecModel.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, waveform_24khz: torch.Tensor) -> torch.Tensor:
        """waveform_24khz: (batch, num_samples) at 24 kHz. Returns the quantized (post-RVQ)
        representation (batch, 128, num_frames) -- "the EnCodec latent" throughout TASK-10,
        matching `FrozenDacEncoder.encode()`'s role for DAC."""
        return self.encode_stages(waveform_24khz).quantized_representation

    @torch.inference_mode()
    def encode_stages(self, waveform_24khz: torch.Tensor) -> EncodecEncodeStages:
        """waveform_24khz: (batch, num_samples) at 24 kHz. Returns the pre-quantizer embedding,
        the post-RVQ quantized representation, and per-stage `audio_codes` from a single forward
        pass -- mirrors `FrozenDacEncoder.encode_stages`."""
        assert waveform_24khz.ndim == 2, "expected (batch, num_samples)"
        input_values = waveform_24khz.unsqueeze(1).to(self.device)
        pre_quantizer_encoding = self.model.encoder(input_values)
        audio_codes = self.model.quantizer.encode(pre_quantizer_encoding, self.bandwidth)  # (num_quantizers, batch, frames)
        quantized_representation = self.model.quantizer.decode(audio_codes)
        return EncodecEncodeStages(
            pre_quantizer_encoding=pre_quantizer_encoding,
            quantized_representation=quantized_representation,
            audio_codes=audio_codes.transpose(0, 1),  # -> (batch, num_quantizers, frames)
        )

    @torch.inference_mode()
    def decode(self, quantized_representation: torch.Tensor) -> torch.Tensor:
        """quantized_representation: (batch, 128, num_frames) -> waveform (batch, num_samples)."""
        output = self.model.decoder(quantized_representation.to(self.device))
        return output.squeeze(1)
