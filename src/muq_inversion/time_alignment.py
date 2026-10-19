"""Physical time alignment between MuQ tokens and BigVGAN-v2 mel frames (TASK-16).

TASK-15 resized the MuQ sequence to the mel length with `F.interpolate(align_corners=False)`,
which implicitly places token i at (i + 0.5) MuQ hops. The TASK-15 audit measured the actual
centers (MuQ token i at 960*i samples, BigVGAN frame j at 256*j + 128) and found the mel frames
were conditioned on MuQ content ~15-20 ms early. This module derives both sets of centers from
the loaded implementations and interpolates MuQ features at the exact time of each mel frame.

Centers are in samples of the 24 kHz waveform both models consume.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _conv_time_center_map(conv: nn.Conv2d) -> tuple[float, float]:
    """Affine map (a, b) such that output time index k is centered at input index a*k + b, for a
    Conv2d whose last axis is time (MuQ's Conv2dSubsampling layout is (b, c, freq, time))."""
    k, s, p, d = conv.kernel_size[1], conv.stride[1], conv.padding[1], conv.dilation[1]
    return float(s), float(-p + d * (k - 1) / 2)


def _compose(outer: tuple[float, float], inner: tuple[float, float]) -> tuple[float, float]:
    """inner maps output->middle, outer maps middle->input; returns output->input."""
    a1, b1 = outer
    a2, b2 = inner
    return a1 * a2, a1 * b2 + b1


def muq_token_centers(muq_model: nn.Module, num_tokens: int) -> np.ndarray:
    """Sample position of each MuQ `last_hidden_state` token's center, derived from the loaded
    model: the mel STFT front end (hop, centering, padding) composed with the time-axis geometry
    of every convolution in `Conv2dSubsampling`. `muq_model` is the inner `MuQModel`
    (`FrozenMuQEncoder(...).model.model`). The conformer that follows doesn't change the sequence
    length or position, so token i keeps the conv front end's center.

    Each `Res2dModule` has a main path (conv1 -> conv2) and, when it downsamples, a shortcut
    (conv3); both must map to the same centers or the token time would be ambiguous."""
    spec = muq_model.preprocessor_melspec_2048.mel_stft.spectrogram
    hop = spec.hop_length
    # torch.stft(center=True) pads n_fft//2 so frame k is centered at k*hop; extra `pad` shifts left.
    frame0 = 0.0 if spec.center else spec.n_fft / 2
    frame0 -= spec.pad

    total = (1.0, 0.0)  # token index -> mel-frame index
    for block in muq_model.conv.conv:
        main = _compose(_conv_time_center_map(block.conv1), _conv_time_center_map(block.conv2))
        if getattr(block, "diff", False):
            shortcut = _conv_time_center_map(block.conv3)
            if not np.allclose(main, shortcut):
                raise ValueError(f"MuQ Res2dModule main path {main} and shortcut {shortcut} disagree on time centers")
        total = _compose(total, main)
    a, b = total
    return frame0 + hop * (a * np.arange(num_tokens) + b)


def bigvgan_mel_frame_centers(h, num_frames: int) -> np.ndarray:
    """Sample position of each BigVGAN-v2 mel frame's center. Mirrors
    `meldataset.mel_spectrogram` for the checkpoint's `h`: the waveform is reflect-padded by
    (n_fft - hop) // 2 on both sides, then `torch.stft(center=False)` takes frame j from padded
    sample j*hop with an n_fft-long window (a shorter `win_size` window is centered inside it), so
    frame j is centered at j*hop - pad + n_fft/2 in the original waveform. Tests verify this against
    impulse responses of the official function."""
    pad = (h.n_fft - h.hop_size) // 2
    return np.arange(num_frames) * h.hop_size - pad + h.n_fft / 2


def source_positions(target_centers: np.ndarray, source_centers: np.ndarray) -> torch.Tensor:
    """Fractional source index at each target time, by linear interpolation of the source centers.
    Targets before the first or after the last source center map to 0 or len-1 (edge
    replication); there is no source information beyond them."""
    if np.any(np.diff(source_centers) <= 0):
        raise ValueError("source centers must be strictly increasing")
    idx = np.interp(target_centers, source_centers, np.arange(len(source_centers), dtype=np.float64))
    return torch.from_numpy(idx).float()


def interpolate_at_positions(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """x: (B, C, T_src); positions: (T_tgt,) fractional indices in [0, T_src - 1].
    Returns (B, C, T_tgt), linear interpolation between neighboring source frames."""
    if positions.min() < 0 or positions.max() > x.shape[-1] - 1:
        raise ValueError(f"positions must lie in [0, {x.shape[-1] - 1}]")
    pos = positions.to(x.device, x.dtype)
    lo = pos.floor().long().clamp(max=x.shape[-1] - 1)
    hi = (lo + 1).clamp(max=x.shape[-1] - 1)
    w = pos - lo.to(pos.dtype)
    return x[..., lo] * (1 - w) + x[..., hi] * w
