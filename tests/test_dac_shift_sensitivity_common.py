"""Tests for tasks/TASK-07.md's DAC shift/polarity sensitivity diagnostic -- the pure helpers in
`muq_inversion.dac_shift_sensitivity_common`.
"""

from __future__ import annotations

import csv

import pytest
import torch

from muq_inversion.dac_shift_sensitivity_common import (
    codebook_agreement,
    cosine_similarity_per_frame,
    sample_cross_track_pairs,
    select_stratified_tracks,
)


def _write_manifest(path, rows):
    fieldnames = ["track_id", "genre", "relative_path", "split"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_select_stratified_tracks_picks_first_n_per_genre_ascending(tmp_path):
    rows = []
    for genre in ["blues", "jazz"]:
        for i in range(3):
            rows.append({"track_id": f"{genre}.{i:05d}", "genre": genre, "relative_path": "x", "split": "val"})
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)

    selected = select_stratified_tracks(manifest, split="val", tracks_per_genre=2)
    assert [t.track_id for t in selected] == ["blues.00000", "blues.00001", "jazz.00000", "jazz.00001"]


def test_select_stratified_tracks_ignores_other_splits(tmp_path):
    rows = [
        {"track_id": "blues.00000", "genre": "blues", "relative_path": "x", "split": "train"},
        {"track_id": "blues.00001", "genre": "blues", "relative_path": "x", "split": "val"},
        {"track_id": "blues.00002", "genre": "blues", "relative_path": "x", "split": "val"},
    ]
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)

    selected = select_stratified_tracks(manifest, split="val", tracks_per_genre=2)
    assert [t.track_id for t in selected] == ["blues.00001", "blues.00002"]


def test_select_stratified_tracks_raises_if_genre_too_small(tmp_path):
    rows = [{"track_id": "blues.00000", "genre": "blues", "relative_path": "x", "split": "val"}]
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="only 1"):
        select_stratified_tracks(manifest, split="val", tracks_per_genre=2)


def test_cosine_similarity_per_frame_identical_is_one():
    a = torch.randn(1, 1024, 10)
    assert cosine_similarity_per_frame(a, a, frame_dim=2) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_per_frame_frames_first_layout():
    a = torch.randn(1, 10, 128)
    assert cosine_similarity_per_frame(a, a, frame_dim=1) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_per_frame_orthogonal_is_zero():
    a = torch.zeros(1, 2, 4)
    a[0, 0, :] = 1.0
    b = torch.zeros(1, 2, 4)
    b[0, 1, :] = 1.0
    assert cosine_similarity_per_frame(a, b, frame_dim=2) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_per_frame_shape_mismatch_raises():
    a = torch.randn(1, 1024, 10)
    b = torch.randn(1, 1024, 9)
    with pytest.raises(ValueError, match="shape mismatch"):
        cosine_similarity_per_frame(a, b, frame_dim=2)


def test_codebook_agreement_identical_is_one():
    codes = torch.randint(0, 1024, (2, 32, 15))
    agreement = codebook_agreement(codes, codes)
    assert agreement["stage0_agreement"] == pytest.approx(1.0)
    assert agreement["mean_agreement_all_stages"] == pytest.approx(1.0)


def test_codebook_agreement_partial_match():
    codes_a = torch.zeros(1, 2, 4, dtype=torch.long)
    codes_b = torch.zeros(1, 2, 4, dtype=torch.long)
    codes_b[0, 0, :2] = 1  # stage 0: 2 of 4 frames disagree
    agreement = codebook_agreement(codes_a, codes_b)
    assert agreement["stage0_agreement"] == pytest.approx(0.5)
    assert agreement["mean_agreement_all_stages"] == pytest.approx(0.75)


def test_sample_cross_track_pairs_never_same_track():
    track_of_crop = {"a1": "trackA", "a2": "trackA", "b1": "trackB", "c1": "trackC"}
    pairs = sample_cross_track_pairs(list(track_of_crop), track_of_crop, n_pairs=3, seed=0)
    assert len(pairs) == 3
    for x, y in pairs:
        assert track_of_crop[x] != track_of_crop[y]


def test_sample_cross_track_pairs_deterministic():
    track_of_crop = {f"c{i}": f"track{i % 5}" for i in range(20)}
    pairs1 = sample_cross_track_pairs(list(track_of_crop), track_of_crop, n_pairs=10, seed=0)
    pairs2 = sample_cross_track_pairs(list(track_of_crop), track_of_crop, n_pairs=10, seed=0)
    assert pairs1 == pairs2


def test_sample_cross_track_pairs_raises_if_too_few():
    track_of_crop = {"a1": "trackA", "a2": "trackA"}
    with pytest.raises(ValueError, match="only 0"):
        sample_cross_track_pairs(list(track_of_crop), track_of_crop, n_pairs=1, seed=0)
