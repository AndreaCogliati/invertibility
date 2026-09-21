"""Pure, testable helpers for `tasks/TASK-07.md`'s DAC shift/polarity sensitivity diagnostic
(`scripts/analyze_dac_shift_sensitivity.py`).

Kept separate from the script, mirroring this project's existing convention (`gate4_common.py`,
`gate3_spectrogram_common.py`) -- so track selection, frame-alignment, and similarity/agreement
logic can be unit-tested without loading any model or audio file.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
import random
from pathlib import Path

import torch


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
