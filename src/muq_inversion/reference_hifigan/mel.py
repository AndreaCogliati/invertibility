"""Vendored from https://github.com/jik876/hifi-gan/blob/master/meldataset.py (MIT License).

`mel_spectrogram` kept functionally identical: `librosa.filters.mel` filterbank (not
`torchaudio.transforms.MelSpectrogram` -- a different implementation with different defaults),
manual reflect-pad by `(n_fft - hop_size) // 2` each side, `torch.stft` with `center=False`,
magnitude via `sqrt(power_sum + 1e-9)`, natural-log compression with `clip_val=1e-5`.

One required compatibility fix, not a design change: the original calls
`librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)` positionally, matching an older
librosa API. This project's installed `librosa==1.0.0` makes every parameter of
`librosa.filters.mel` keyword-only (confirmed via `inspect.signature` before writing this) --
calling it positionally raises a `TypeError`. Fixed by passing everything as keywords; the
filterbank produced is otherwise identical (same `htk=False`, `norm='slaney'` defaults the
original repo relied on).
"""

from __future__ import annotations

import torch
from librosa.filters import mel as librosa_mel_fn

_mel_basis_cache: dict[str, torch.Tensor] = {}
_hann_window_cache: dict[str, torch.Tensor] = {}


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float | None,
) -> torch.Tensor:
    """y: (batch, num_samples) in [-1, 1] -> log-mel spectrogram (batch, num_mels, num_frames)."""
    cache_key = f"{fmax}_{num_mels}_{n_fft}_{sampling_rate}_{fmin}_{y.device}"
    if cache_key not in _mel_basis_cache:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis_cache[cache_key] = torch.from_numpy(mel).float().to(y.device)
        _hann_window_cache[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y_padded = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y_padded,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_hann_window_cache[str(y.device)],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    spec = torch.matmul(_mel_basis_cache[cache_key], spec)
    return torch.log(torch.clamp(spec, min=1e-5))
