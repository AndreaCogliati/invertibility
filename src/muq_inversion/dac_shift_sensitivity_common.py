"""Pure, testable helpers for `tasks/TASK-07.md`'s DAC shift/polarity sensitivity diagnostic
(`scripts/analyze_dac_shift_sensitivity.py`), its audio-domain extension (`tasks/TASK-08.md`,
`scripts/analyze_dac_latent_vs_audio_shift.py`), and the Optimal Transport geometry diagnostic
built on both (`tasks/TASK-09.md`, `scripts/analyze_dac_ot_geometry.py`).

Kept separate from the scripts, mirroring this project's existing convention (`gate4_common.py`,
`gate3_spectrogram_common.py`) -- so track selection, frame-alignment, and similarity/agreement
logic can be unit-tested without loading any model or audio file.

`load_padded_crops`/`slice_shifted`/`crop_id` below generalize TASK-07's script-local versions of
the same logic (parameterized by `max_shift` instead of a module constant) so TASK-08 can reuse
identical crop-extraction behavior at a larger margin (1024 vs. TASK-07's 256 samples) without
duplicating or modifying TASK-07's own script, per TASK-08's "do not modify any existing
experiment" instruction.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
import random
from pathlib import Path

import numpy as np
import ot
import torch
from scipy import stats

from muq_inversion.audio import load_resample_validate


@dataclasses.dataclass(frozen=True)
class TrackSpec:
    track_id: str
    genre: str
    relative_path: str


def select_stratified_tracks(manifest_path: Path, split: str, tracks_per_genre: int) -> list[TrackSpec]:
    """Deterministic genre-stratified track selection: for `split`, group by genre, sort each
    group by `track_id` ascending, take the first `tracks_per_genre` of each. No randomness --
    reproducible by construction, matching this project's existing manifest-building convention
    (`build_gate3_scale_manifests.py`'s round-robin selection). Raises if any genre has fewer than
    `tracks_per_genre` tracks in `split`, rather than silently returning short."""
    with open(manifest_path, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == split]

    by_genre: dict[str, list[TrackSpec]] = {}
    for row in rows:
        by_genre.setdefault(row["genre"], []).append(
            TrackSpec(track_id=row["track_id"], genre=row["genre"], relative_path=row["relative_path"])
        )

    selected: list[TrackSpec] = []
    for genre in sorted(by_genre):
        candidates = sorted(by_genre[genre], key=lambda t: t.track_id)
        if len(candidates) < tracks_per_genre:
            raise ValueError(
                f"genre {genre!r} has only {len(candidates)} {split}-split tracks, need {tracks_per_genre}"
            )
        selected.extend(candidates[:tracks_per_genre])
    return selected


def cosine_similarity_per_frame(a: torch.Tensor, b: torch.Tensor, frame_dim: int) -> float:
    """Mean cosine similarity between `a` and `b` across all frames, for either DAC's
    channels-first (batch, channels, frames) layout or MuQ/log-mel's frames-first
    (batch, frames, channels) layout -- `frame_dim` says which axis is frames so the channel axis
    (the one `cosine_similarity` reduces over) is always the other one. Shapes must match
    exactly (same batch and frame count) -- this is an equal-length frame-index-aligned
    comparison, not a cross-correlation search."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    channel_dim = 1 if frame_dim == 2 else -1
    per_frame = torch.nn.functional.cosine_similarity(a, b, dim=channel_dim)
    return float(per_frame.mean().item())


def codebook_agreement(codes_a: torch.Tensor, codes_b: torch.Tensor) -> dict[str, float]:
    """`codes_a`/`codes_b`: (batch, n_codebooks, num_frames) integer codebook indices. Returns the
    fraction of (batch, frame) positions where stage-0's index matches, and the same averaged
    over all `n_codebooks` stages -- `tasks/TASK-07.md`'s exact "stage 0 only, and mean over all
    32 stages" requirement (32 is this checkpoint's measured `n_codebooks`, not assumed here)."""
    if codes_a.shape != codes_b.shape:
        raise ValueError(f"shape mismatch: {tuple(codes_a.shape)} vs {tuple(codes_b.shape)}")
    match = codes_a == codes_b  # (batch, n_codebooks, num_frames)
    stage0 = float(match[:, 0, :].float().mean().item())
    mean_all_stages = float(match.float().mean().item())
    return {"stage0_agreement": stage0, "mean_agreement_all_stages": mean_all_stages}


def sample_cross_track_pairs(crop_ids: list[str], track_of_crop: dict[str, str], n_pairs: int, seed: int) -> list[tuple[str, str]]:
    """Deterministically samples `n_pairs` distinct (crop_id, crop_id) pairs from `crop_ids` whose
    underlying tracks differ (`tasks/TASK-07.md`'s "unrelated crops" random-pair baseline,
    "computed once"). Sampling is without replacement over all cross-track pairs, seeded for
    reproducibility. Raises if fewer than `n_pairs` cross-track pairs exist."""
    all_pairs = [
        (a, b) for a, b in itertools.combinations(crop_ids, 2) if track_of_crop[a] != track_of_crop[b]
    ]
    if len(all_pairs) < n_pairs:
        raise ValueError(f"only {len(all_pairs)} cross-track pairs available, need {n_pairs}")
    rng = random.Random(seed)
    return rng.sample(all_pairs, n_pairs)


def crop_id(track_id: str, crop_start: float) -> str:
    return f"{track_id}_t{crop_start:.1f}s"


def slice_shifted(buffer: torch.Tensor, shift: int, crop_samples: int, max_shift: int) -> torch.Tensor:
    """buffer: (crop_samples + 2*max_shift,), a padded real-audio window (see `load_padded_crops`).
    shift=0 is the canonical crop; a nonzero `shift` re-slices the same buffer `shift` samples
    later (positive) or earlier (negative) -- never `torch.roll`/wraparound, matching TASK-07's
    "real-audio margin, not circular wraparound" design."""
    start = max_shift + shift
    return buffer[start : start + crop_samples]


def load_padded_crops(
    tracks: list[TrackSpec],
    raw_audio_root: Path,
    sample_rate: int,
    crop_start_seconds: list[float],
    crop_samples: int,
    max_shift: int,
) -> tuple[dict[str, torch.Tensor], dict[str, str], list[str]]:
    """Returns (crop_id -> padded buffer of length crop_samples+2*max_shift, crop_id -> track_id,
    list of dropped crop_ids) -- a failing (non-finite) crop is dropped and logged, not fatal,
    matching TASK-07's stop-condition-#2 policy. Generalizes TASK-07's script-local
    `load_padded_crops` by taking `max_shift` as a parameter instead of a module constant, so a
    caller needing a larger margin (TASK-08's shift schedule extends to 1024 samples, vs.
    TASK-07's 256) gets identical crop-extraction behavior without duplicating or modifying
    TASK-07's own script."""
    buffers: dict[str, torch.Tensor] = {}
    track_of_crop: dict[str, str] = {}
    dropped: list[str] = []
    for track in tracks:
        waveform, _orig_sr = load_resample_validate(raw_audio_root / track.relative_path, sample_rate)
        for crop_start in crop_start_seconds:
            cid = crop_id(track.track_id, crop_start)
            start = round(crop_start * sample_rate)
            lo, hi = start - max_shift, start + crop_samples + max_shift
            if lo < 0 or hi > len(waveform):
                raise ValueError(
                    f"STOP CONDITION: {track.track_id} crop at {crop_start}s needs samples "
                    f"[{lo},{hi}) but track has only {len(waveform)} resampled samples"
                )
            buf = waveform[lo:hi]
            if not np.all(np.isfinite(buf)):
                print(f"dropping {cid}: non-finite samples in padded buffer")
                dropped.append(cid)
                continue
            buffers[cid] = torch.from_numpy(buf.copy()).float()
            track_of_crop[cid] = track.track_id
    return buffers, track_of_crop, dropped


def align_shifted_pair(a: torch.Tensor, b: torch.Tensor, shift: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`tasks/TASK-08.md`'s required audio-alignment correction: `a` and `b` are two equal-length
    1-D signals (e.g. DAC reconstructions), where `b`'s content started `shift` samples later in
    absolute source-time than `a`'s (matching this module's `slice_shifted` sign convention).
    Crops the non-overlapping edge from whichever signal leads, returning two equal-length
    excerpts that cover the exact same underlying source-time span -- so a similarity metric on
    the result reflects content difference, not the trivial fact that the two signals are offset
    copies of (almost) the same audio. `shift=0` returns both signals unchanged. Raises if the
    signals have no overlap left (`abs(shift) >= len(a)`)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    n = a.shape[-1]
    if abs(shift) >= n:
        raise ValueError(f"shift {shift} leaves no overlap for length-{n} signals")
    if shift == 0:
        return a, b
    overlap = n - abs(shift)
    if shift > 0:
        return a[..., shift : shift + overlap], b[..., 0:overlap]
    k = -shift
    return a[..., 0:overlap], b[..., k : k + overlap]


def l2_distance_per_frame(a: torch.Tensor, b: torch.Tensor, frame_dim: int) -> float:
    """Mean L2 (Euclidean) distance between `a` and `b` across all frames -- the literal
    "Log-Mel Distance" metric `tasks/TASK-08.md` requires, alongside `cosine_similarity_per_frame`
    (used for the same pair when a similarity, not a distance, is needed for a figure/correlation
    axis). Same channels-first-vs-frames-first `frame_dim` convention as
    `cosine_similarity_per_frame`."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    channel_dim = 1 if frame_dim == 2 else -1
    per_frame = torch.linalg.vector_norm(a - b, dim=channel_dim)
    return float(per_frame.mean().item())


def correlate_metrics(x: list[float], y: list[float]) -> dict[str, float]:
    """Pearson and Spearman correlation between two equal-length metric sequences --
    `tasks/TASK-08.md`'s required "Additional Analysis" (latent similarity vs. audio similarity,
    across all crop/shift points). Thin wrapper over `scipy.stats` (already a project dependency,
    per this task's "do not add a new dependency" instruction) so the call site stays a single
    line and the result is directly JSON-serializable (plain floats, not scipy result objects)."""
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def sinkhorn_epsilon_for_cloud(points: np.ndarray, fraction: float) -> float:
    """`tasks/TASK-09.md`'s required documented epsilon choice: `fraction` times the median of
    the nonzero pairwise squared-Euclidean distances within `points` (a single (num_frames,
    num_channels) point cloud -- typically the canonical, shift-0 latent). Scaling the
    regularization to the cloud's own distance scale, rather than a fixed absolute epsilon,
    keeps Sinkhorn well-conditioned across latents of very different magnitude without any
    per-pair manual tuning -- verified empirically (this module's tests, and this task's own
    real-DAC-latent trial runs) to converge cleanly at `fraction=0.1` where a smaller fraction
    (e.g. 0.05) can fail to converge within the default iteration budget."""
    pairwise = ot.dist(points, points, metric="sqeuclidean")
    nonzero = pairwise[pairwise > 0]
    if nonzero.size == 0:
        return 1e-6
    return max(float(fraction * np.median(nonzero)), 1e-6)


def sinkhorn_ot_divergence(z0: np.ndarray, zk: np.ndarray, epsilon: float, num_iter_max: int = 5000) -> float:
    """Debiased entropic-regularized Sinkhorn divergence between two point clouds `z0`, `zk`
    (each (num_frames, num_channels), uniform mass per frame) -- `tasks/TASK-09.md`'s "Optimal
    Transport distance" metric. Uses POT's `ot.bregman.empirical_sinkhorn_divergence` (squared-
    Euclidean ground cost) rather than the plain entropic OT cost (`sinkhorn2`): the plain cost
    is verified (this module's own tests) to NOT vanish for `z0 == zk` -- it carries an O(epsilon)
    entropy bias even at zero true distance -- while the debiased divergence subtracts the two
    self-transport terms and is confirmed to return exactly 0.0 for identical clouds, giving an
    interpretable zero-point that matches MSE=0/cosine=1 at zero shift."""
    return float(ot.bregman.empirical_sinkhorn_divergence(z0, zk, reg=epsilon, numIterMax=num_iter_max))
