"""Latent-space norm/variance/autocorrelation diagnostics required by `tasks/TASK-01.md`.

Pure numpy functions, unit-tested independently of any model or cache. The exact statistics
(`vector_norm_mean`, `channel variance mean`, `lag1_autocorr`) mirror
`scripts/analyze_dac_manifold.py`'s ad hoc versions of the same computations -- pulled out here
into a shared, tested module rather than duplicated a third time.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class LatentStats:
    vector_norm_mean: float
    vector_norm_median: float
    vector_norm_std: float
    channel_variance_mean: float
    temporal_variance_mean: float
    frame_to_frame_variance_mean: float
    lag1_autocorr: float


def compute_latent_stats(frames: np.ndarray) -> LatentStats:
    """frames: (total_frames, channels) -- typically concatenated across crops within a split.

    - `vector_norm_mean`/`median`/`std`: L2 norm of each frame vector (median added for
      `tasks/TASK-03.md`'s explicit "median latent/prediction norm" requirement -- more robust
      than the mean to the occasional huge-but-finite outlier this pipeline has shown, e.g. the
      val-MSE spikes to 1e3-1e8 seen mid-training in [[spectrogram-positive-control]] and
      [[dense-spectrogram-positive-control]]).
    - `channel_variance_mean`: per-channel variance across frames, averaged over channels
      (collapses to ~0 under channel-wise norm collapse).
    - `temporal_variance_mean`: variance of each channel's value across time, averaged over
      channels -- same computation as `channel_variance_mean` when `frames` spans one crop's
      timeline; kept as a separate named field because the task's diagnostic spec lists "temporal
      variance" and "latent variance ratio" as distinct required items.
    - `frame_to_frame_variance_mean`: variance of consecutive-frame differences, averaged over
      channels -- a second, complementary over-smoothing signal to `lag1_autocorr` (TASK-03's
      "frame-to-frame variance" requirement): near-zero under extreme over-smoothing regardless of
      the autocorrelation value's sign convention.
    - `lag1_autocorr`: lag-1 autocorrelation of the (channel-mean-centered) frame sequence,
      pooled over all channels -- high values indicate over-smoothing / excessive temporal
      redundancy.
    """
    if frames.ndim != 2:
        raise ValueError(f"expected (frames, channels), got shape {frames.shape}")
    norms = np.linalg.norm(frames, axis=1)
    channel_var = frames.var(axis=0)
    centered = frames - frames.mean(axis=0, keepdims=True)
    if frames.shape[0] < 2:
        lag1 = float("nan")
        frame_to_frame_var = float("nan")
    else:
        num = np.sum(centered[:-1] * centered[1:])
        den = np.sum(centered[:-1] ** 2) + 1e-9
        lag1 = float(num / den)
        frame_to_frame_var = float(np.diff(frames, axis=0).var(axis=0).mean())
    return LatentStats(
        vector_norm_mean=float(norms.mean()),
        vector_norm_median=float(np.median(norms)),
        vector_norm_std=float(norms.std()),
        channel_variance_mean=float(channel_var.mean()),
        temporal_variance_mean=float(channel_var.mean()),
        frame_to_frame_variance_mean=frame_to_frame_var,
        lag1_autocorr=lag1,
    )


def ratio(predicted: LatentStats, oracle: LatentStats) -> dict[str, float]:
    """Predicted-to-oracle ratios: 1.0 = matched, far from 1.0 = collapse/inflation."""
    eps = 1e-12
    return {
        "norm_ratio": predicted.vector_norm_mean / (oracle.vector_norm_mean + eps),
        "variance_ratio": predicted.channel_variance_mean / (oracle.channel_variance_mean + eps),
        "temporal_variance_ratio": predicted.temporal_variance_mean / (oracle.temporal_variance_mean + eps),
        "lag1_autocorr_predicted": predicted.lag1_autocorr,
        "lag1_autocorr_oracle": oracle.lag1_autocorr,
    }
