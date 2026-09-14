"""Log-mel spectrogram representation for the Spectrogram-to-DAC positive-control experiment
(`tasks/TASK-01.md`).

Mirrors `encoder.py` / `dac_encoder.py`'s "measure, don't assume" discipline even though this
representation is a deterministic transform, not a learned/frozen model: frame geometry is
asserted against DAC's own measured frame rate at call time, not just documented in a comment.

## Parameter choice, justified for this experiment (not copied from a speech recipe)

- `hop_length=320`: exactly DAC's own hop (`24000 / 75 Hz`, see `dac_encoder.py`'s measured
  frame rate) -- chosen so the spectrogram lands on DAC's temporal grid with no upsampling or
  repeat-interleave step, unlike MuQ's 25 Hz representation (which needs `repeat_interleave(3)`
  to reach DAC's 75 Hz). This removes one whole alignment step relative to the MuQ pipeline.
- `n_fft=1024`, `win_length=1024`: standard ~42.7ms analysis window at 24kHz, wide enough for
  low-frequency mel bins to be meaningful at this hop size.
- `n_mels=128`: this task's suggested starting point.
- `f_min=0`, `f_max=12000`: full range up to Nyquist at 24kHz -- no rolloff assumption.
- `power=1.0` (magnitude, not power-squared) then `log(clamp_min(eps))`: matches this codebase's
  existing `MelSpectrogramLoss` convention (`models/losses.py`) for consistency, not a new
  normalization scheme. No additional per-channel or per-clip normalization is applied -- log
  compression alone, the minimal standard preprocessing for a spectrogram input (distinct from
  the DAC latent *target*, which this experiment leaves fully unnormalized per its own
  instructions).

## Temporal alignment

`torchaudio.transforms.MelSpectrogram` defaults to `center=True` (reflect-padded analysis
windows), which produces `1 + num_samples // hop_length` frames -- one frame more than DAC's
`num_samples // hop_length` for every crop length used in this experiment (measured: 151 vs. 150
for a 2.0s/48000-sample crop). The final centered frame extends past the crop's last real sample
under that padding, so it is dropped: `align_to_dac_frames` truncates to DAC's exact frame count
and asserts the pre-truncation excess is exactly 0 or 1 frame, raising otherwise (a larger
mismatch would indicate a parameter or sample-rate error, not expected reflect-padding overhang).
"""

from __future__ import annotations

import dataclasses

import torch
import torchaudio

SAMPLE_RATE = 24000
N_FFT = 1024
HOP_LENGTH = 320  # == 24000 / 75, DAC's measured frame rate -- see module docstring
WIN_LENGTH = 1024
N_MELS = 128
F_MIN = 0.0
F_MAX = 12000.0
LOG_EPS = 1e-5


@dataclasses.dataclass(frozen=True)
class SpectrogramProbeResult:
    waveform_shape: tuple[int, ...]
    feature_shape: tuple[int, ...]
    channel_dim: int
    frames_per_second: float
    samples_per_frame: float
    dtype: str
    device: str
    is_finite: bool


class LogMelSpectrogramEncoder:
    """Deterministic (not learned) log-mel feature extractor, used as a positive-control
    replacement for `FrozenMuQEncoder` in this experiment only."""

    def __init__(self, device: torch.device):
        self.device = device
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=1.0,
        ).to(device)

    @torch.inference_mode()
    def encode(self, waveform_24khz: torch.Tensor) -> torch.Tensor:
        """waveform_24khz: (batch, num_samples) at 24 kHz. Returns log-mel features
        (batch, num_frames, n_mels) -- frames-first, matching MuQ's `last_hidden_state` layout
        (unlike DAC's channels-first `quantized_representation`)."""
        assert waveform_24khz.ndim == 2, "expected (batch, num_samples)"
        mel = self.mel(waveform_24khz.to(self.device))  # (batch, n_mels, num_frames)
        log_mel = torch.log(mel.clamp_min(LOG_EPS))
        return log_mel.transpose(1, 2)  # -> (batch, num_frames, n_mels)

    @torch.inference_mode()
    def probe(self, waveform_24khz: torch.Tensor, sample_rate: int) -> SpectrogramProbeResult:
        features = self.encode(waveform_24khz)
        duration = waveform_24khz.shape[-1] / sample_rate
        num_frames = features.shape[1]
        return SpectrogramProbeResult(
            waveform_shape=tuple(waveform_24khz.shape),
            feature_shape=tuple(features.shape),
            channel_dim=features.shape[-1],
            frames_per_second=num_frames / duration if duration > 0 else float("nan"),
            samples_per_frame=waveform_24khz.shape[-1] / num_frames if num_frames > 0 else float("nan"),
            dtype=str(features.dtype),
            device=str(features.device),
            is_finite=bool(torch.isfinite(features).all().item()),
        )


def align_to_dac_frames(spectrogram: torch.Tensor, dac_num_frames: int) -> torch.Tensor:
    """Truncates `spectrogram` (batch, num_frames, n_mels) to exactly `dac_num_frames` frames.

    Only ever drops trailing frames (never pads) and only for a 0- or 1-frame excess -- see
    module docstring. Raises `ValueError` for any larger mismatch, since that would signal a
    hop-length/sample-rate configuration error rather than expected edge padding.
    """
    excess = spectrogram.shape[1] - dac_num_frames
    if excess not in (0, 1):
        raise ValueError(
            f"STOP CONDITION: spectrogram frame count {spectrogram.shape[1]} does not match "
            f"DAC's {dac_num_frames} within the expected 0-1 frame padding overhang (excess={excess})"
        )
    return spectrogram[:, :dac_num_frames, :]
