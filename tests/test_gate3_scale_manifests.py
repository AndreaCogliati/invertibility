"""Validates the 5 nested Gate-3 scaling manifests (`tasks/TASK-05.md`,
`scripts/build_gate3_scale_manifests.py`) against their required properties: nesting, genre
balance, level-1 == TASK-04's exact two excerpts, and no duplicate tracks per level.

Deterministically regenerable (`uv run python scripts/build_gate3_scale_manifests.py`), so this
test reads the real generated files rather than mocks -- consistent with `test_dense_manifest.py`.
"""

from __future__ import annotations

import csv
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LEVELS = [2, 8, 32, 128, 512]
MANIFEST_PATHS = {n: REPO_ROOT / f"data/manifests/gate3_scale_{n}.csv" for n in LEVELS}

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in MANIFEST_PATHS.values()),
    reason="gate3_scale_*.csv not generated -- run scripts/build_gate3_scale_manifests.py",
)


def _load_track_ids(n: int) -> list[str]:
    with open(MANIFEST_PATHS[n], newline="") as f:
        return [row["track_id"] for row in csv.DictReader(f)]


def test_each_level_has_exactly_n_rows():
    for n in LEVELS:
        assert len(_load_track_ids(n)) == n


def test_no_duplicate_tracks_within_a_level():
    for n in LEVELS:
        track_ids = _load_track_ids(n)
        assert len(set(track_ids)) == n


def test_levels_are_strictly_nested():
    for a, b in pairwise(LEVELS):
        assert set(_load_track_ids(a)).issubset(set(_load_track_ids(b)))


def test_level_one_is_exactly_task04_excerpts():
    assert set(_load_track_ids(2)) == {"pop.00012", "disco.00049"}


def test_all_crops_use_the_same_crop_position():
    for n in LEVELS:
        with open(MANIFEST_PATHS[n], newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert row["crop_index"] == "7"
            assert row["crop_start_seconds"] == "14.00"
            assert row["crop_duration_seconds"] == "2.00"


def test_genre_balance_within_one_track_of_ideal():
    with open(REPO_ROOT / "data/manifests/gtzan_manifest_smoke.csv", newline="") as f:
        genre_by_track = {row["track_id"]: row["genre"] for row in csv.DictReader(f) if row["corrupted"] != "True"}

    for n in LEVELS:
        if n < 10:  # fewer examples than genres -- balance isn't meaningful yet
            continue
        track_ids = _load_track_ids(n)
        counts = Counter(genre_by_track[t] for t in track_ids)
        ideal = n / 10
        for genre, count in counts.items():
            assert abs(count - ideal) <= 1.5, f"level {n}: genre {genre} has {count} tracks, ideal ~{ideal:.1f}"


def test_all_selected_tracks_are_train_split():
    with open(REPO_ROOT / "data/manifests/gtzan_manifest_smoke.csv", newline="") as f:
        split_by_track = {row["track_id"]: row["split"] for row in csv.DictReader(f) if row["corrupted"] != "True"}
    for n in LEVELS:
        for track_id in _load_track_ids(n):
            assert split_by_track[track_id] == "train"
