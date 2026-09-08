"""Ground-truth mel-spectrogram "encoder" (experiment/mel-hifigan-sanity branch only).

Not part of the MuQ baseline -- see notes/DECISIONS.md#mel-hifigan-sanity for why this branch
exists (a sanity check: can the current HiFi-GAN-style generator/discriminator/training recipe
memorize two clips at all when conditioned on the exact representation HiFi-GAN was originally
designed to consume, before blaming MuQ or EnCodec for any remaining reconstruction gap?).

There is no learned upstream model here -- the "encoder" is a deterministic log-mel-spectrogram
transform applied directly to the target waveform crop itself (the classic vocoder "copy
synthesis" sanity check). Kept as a class with the same `encode()`/`probe()` interface as
`FrozenMuQEncoder`/`FrozenEncodecEncoder` so `train.py`'s training loop can treat all three
conditioning sources polymorphically without special-casing this one.
"""

from __future__ import annotations

import torch
import torchaudio

from .config import GroundTruthMelConfig
from .encoder import MuQProbeResult, measure_feature_geometry


class GroundTruthMelEncoder:
    """Computes a log-mel spectrogram directly from the input waveform. Nothing to freeze (no
    learned parameters), but kept `eval()`-only/no-grad for symmetry with the other encoders.

    Uses HiFi-GAN's own standard mel-extraction convention -- manually reflect-pad the waveform
    by `(n_fft - hop_length) // 2` on each side, then run the STFT with `center=False` -- rather
    than `torchaudio`'s default `center=True` framing. `center=True` pads internally and yields
    `1 + num_samples // hop_length` frames (161 for a 48000-sample/hop=300 crop in this project),
    which does not exactly divide `crop_samples`; the manual-pad + `center=False` convention gives
    exactly `num_samples // hop_length` frames (160), matching this project's crop-alignment
    invariant (`crop_samples == num_frames * generator.total_upsample`, see
    notes/DECISIONS.md#gate3-crop-alignment) without any separate cropping/padding step.
    """

    def __init__(self, config: GroundTruthMelConfig, sample_rate: int, device: torch.device):
        self.device = device
        self.eps = 1e-5
        self.pad = (config.n_fft - config.hop_length) // 2
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            f_min=config.fmin,
            f_max=config.fmax,
            power=1.0,
            center=False,
        ).to(device)

    @torch.inference_mode()
    def encode(self, waveform_24khz: torch.Tensor) -> torch.Tensor:
        """waveform_24khz: shape (batch, num_samples) at 24 kHz.

        Returns the log-compressed mel spectrogram, shape (batch, num_frames, n_mels) -- matching
        the (batch, frames, channels) convention used throughout this project. Log-compressed
        (not raw linear-magnitude mel) because that's the standard vocoder-conditioning
        convention, and it keeps the dynamic range comparable to what a real upstream model's
        continuous embedding would look like.
        """
        x = waveform_24khz.to(self.device)
        x_padded = torch.nn.functional.pad(x.unsqueeze(1), (self.pad, self.pad), mode="reflect").squeeze(1)
        mel = self.mel(x_padded)  # (batch, n_mels, num_frames)
        log_mel = torch.log(mel.clamp_min(self.eps))
        return log_mel.transpose(1, 2)

    @torch.inference_mode()
    def probe(self, waveform_24khz: torch.Tensor, sample_rate: int) -> MuQProbeResult:
        """Run one forward pass and measure the feature geometry (Gate-0-style diagnostics)."""
        assert waveform_24khz.ndim == 2, "expected (batch, num_samples)"
        features = self.encode(waveform_24khz)
        return measure_feature_geometry(features, waveform_24khz.shape[-1], sample_rate)
