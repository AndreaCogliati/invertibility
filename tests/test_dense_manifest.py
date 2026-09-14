"""Validates `data/manifests/dense_crops.csv` (built by `scripts/build_dense_gtzan_manifest.py`
for `tasks/TASK-02.md`) against the invariants the training/precompute scripts assume: no crop
exceeds its track's duration, every track is single-split, and the crop count matches the
documented Option-B tiling (non-overlapping 2.0s crops, `floor(duration / 2.0)` per track).

Deterministically regenerable (`uv run python scripts/build_dense_gtzan_manifest.py`), so this
test reads the real generated file rather than a mock -- consistent with this project's existing
manifest-testing style (Gate 1's own committed-manifest checks).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DENSE_MANIFEST = REPO_ROOT / "data/manifests/dense_crops.csv"
SOURCE_MANIFEST = REPO_ROOT / "data/manifests/gtzan_manifest_smoke.csv"

pytestmark = pytest.mark.skipif(
    not DENSE_MANIFEST.exists(), reason="dense_crops.csv not generated -- run scripts/build_dense_gtzan_manifest.py"
)


def _load_dense_rows() -> list[dict]:
    with open(DENSE_MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def _load_source_durations() -> dict[str, float]:
    with open(SOURCE_MANIFEST, newline="") as f:
        return {
            row["track_id"]: float(row["duration_seconds"])
            for row in csv.DictReader(f)
            if row["corrupted"] != "True"
        }


def test_no_crop_exceeds_track_duration():
    durations = _load_source_durations()
    for row in _load_dense_rows():
        start = float(row["crop_start_seconds"])
        dur = float(row["crop_duration_seconds"])
        assert start + dur <= durations[row["track_id"]] + 1e-6, (
            f"{row['track_id']} crop [{start},{start + dur}) exceeds duration {durations[row['track_id']]}"
        )


def test_every_track_is_single_split():
    track_splits: dict[str, set[str]] = {}
    for row in _load_dense_rows():
        track_splits.setdefault(row["track_id"], set()).add(row["split"])
    violations = {t: s for t, s in track_splits.items() if len(s) > 1}
    assert not violations, f"tracks spanning multiple splits: {violations}"


def test_crop_count_matches_floor_division_tiling():
    durations = _load_source_durations()
    crop_counts: dict[str, int] = {}
    for row in _load_dense_rows():
        crop_counts[row["track_id"]] = crop_counts.get(row["track_id"], 0) + 1
    for track_id, duration in durations.items():
        expected = int(duration // 2.0)
        assert crop_counts.get(track_id, 0) == expected, (
            f"{track_id}: expected {expected} crops (duration {duration}), got {crop_counts.get(track_id, 0)}"
        )


def test_crops_are_non_overlapping_and_contiguous_per_track():
    by_track: dict[str, list[tuple[int, float]]] = {}
    for row in _load_dense_rows():
        by_track.setdefault(row["track_id"], []).append(
            (int(row["crop_index"]), float(row["crop_start_seconds"]))
        )
    for track_id, entries in by_track.items():
        entries.sort()
        for i, (crop_index, start) in enumerate(entries):
            assert crop_index == i, f"{track_id}: crop_index gap at position {i}"
            assert start == pytest.approx(i * 2.0), f"{track_id}: crop {i} starts at {start}, expected {i * 2.0}"


def test_dense_manifest_covers_essentially_all_gtzan_tracks():
    durations = _load_source_durations()
    dense_tracks = {row["track_id"] for row in _load_dense_rows()}
    assert dense_tracks == set(durations.keys())
    assert len(dense_tracks) == 999


def test_dense_manifest_is_7to8x_larger_than_task01_manifest():
    """This task's central premise: substantially more crops than TASK-01's 1998 (sparse, 2
    crops/track), while reusing the exact same 999-track split."""
    n = len(_load_dense_rows())
    assert 13000 <= n <= 16000, f"expected roughly 7-8x TASK-01's 1998 crops, got {n}"
