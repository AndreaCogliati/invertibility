"""Tests for tasks/TASK-07.md's DAC shift/polarity sensitivity diagnostic -- the pure helpers in
`muq_inversion.dac_shift_sensitivity_common`.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from muq_inversion.dac_shift_sensitivity_common import (
    align_shifted_pair,
    codebook_agreement,
    correlate_metrics,
    cosine_similarity_per_frame,
    crop_id,
    l2_distance_per_frame,
    sample_cross_track_pairs,
    select_stratified_tracks,
    sinkhorn_epsilon_for_cloud,
    sinkhorn_ot_divergence,
    slice_shifted,
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


def test_crop_id_format():
    assert crop_id("blues.00014", 5.0) == "blues.00014_t5.0s"
    assert crop_id("blues.00014", 15.0) == "blues.00014_t15.0s"


def test_slice_shifted_canonical_is_middle_span():
    buf = torch.arange(20).float()  # crop_samples=10, max_shift=5 -> buf length 20
    canonical = slice_shifted(buf, 0, crop_samples=10, max_shift=5)
    assert torch.equal(canonical, torch.arange(5, 15).float())


def test_slice_shifted_positive_and_negative():
    buf = torch.arange(20).float()
    plus = slice_shifted(buf, 2, crop_samples=10, max_shift=5)
    minus = slice_shifted(buf, -2, crop_samples=10, max_shift=5)
    assert torch.equal(plus, torch.arange(7, 17).float())
    assert torch.equal(minus, torch.arange(3, 13).float())


def test_align_shifted_pair_zero_shift_unchanged():
    a = torch.arange(10).float()
    b = torch.arange(10, 20).float()
    aligned_a, aligned_b = align_shifted_pair(a, b, 0)
    assert torch.equal(aligned_a, a)
    assert torch.equal(aligned_b, b)


def test_align_shifted_pair_positive_shift_drops_correct_edges():
    # b's content started 3 samples later than a's -- a's tail should match b's head.
    a = torch.arange(10).float()
    b = torch.arange(3, 13).float()  # literally a shifted by +3
    aligned_a, aligned_b = align_shifted_pair(a, b, 3)
    assert torch.equal(aligned_a, aligned_b)
    assert len(aligned_a) == 7


def test_align_shifted_pair_negative_shift_drops_correct_edges():
    a = torch.arange(10).float()
    b = torch.arange(-3, 7).float()  # b shifted by -3 relative to a
    aligned_a, aligned_b = align_shifted_pair(a, b, -3)
    assert torch.equal(aligned_a, aligned_b)
    assert len(aligned_a) == 7


def test_align_shifted_pair_shape_mismatch_raises():
    a = torch.zeros(10)
    b = torch.zeros(9)
    with pytest.raises(ValueError, match="shape mismatch"):
        align_shifted_pair(a, b, 1)


def test_align_shifted_pair_no_overlap_raises():
    a = torch.zeros(5)
    b = torch.zeros(5)
    with pytest.raises(ValueError, match="no overlap"):
        align_shifted_pair(a, b, 5)


def test_l2_distance_per_frame_identical_is_zero():
    a = torch.randn(1, 128, 10)
    assert l2_distance_per_frame(a, a, frame_dim=2) == pytest.approx(0.0, abs=1e-6)


def test_l2_distance_per_frame_known_value():
    a = torch.zeros(1, 3, 1)
    b = torch.zeros(1, 3, 1)
    b[0, :, 0] = torch.tensor([3.0, 4.0, 0.0])  # norm = 5
    assert l2_distance_per_frame(a, b, frame_dim=2) == pytest.approx(5.0)


def test_l2_distance_per_frame_shape_mismatch_raises():
    a = torch.randn(1, 128, 10)
    b = torch.randn(1, 128, 9)
    with pytest.raises(ValueError, match="shape mismatch"):
        l2_distance_per_frame(a, b, frame_dim=2)


def test_correlate_metrics_perfect_positive_correlation():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    result = correlate_metrics(x, y)
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_r"] == pytest.approx(1.0)


def test_correlate_metrics_perfect_negative_correlation():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [5.0, 4.0, 3.0, 2.0, 1.0]
    result = correlate_metrics(x, y)
    assert result["pearson_r"] == pytest.approx(-1.0)
    assert result["spearman_r"] == pytest.approx(-1.0)


def test_correlate_metrics_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        correlate_metrics([1.0, 2.0], [1.0])


def test_sinkhorn_epsilon_for_cloud_positive():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(20, 8))
    eps = sinkhorn_epsilon_for_cloud(points, fraction=0.1)
    assert eps > 0


def test_sinkhorn_epsilon_for_cloud_scales_with_fraction():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(20, 8))
    eps_small = sinkhorn_epsilon_for_cloud(points, fraction=0.05)
    eps_big = sinkhorn_epsilon_for_cloud(points, fraction=0.2)
    assert eps_big == pytest.approx(4 * eps_small)


def test_sinkhorn_epsilon_for_cloud_identical_points_floor():
    points = np.zeros((5, 4))
    eps = sinkhorn_epsilon_for_cloud(points, fraction=0.1)
    assert eps == pytest.approx(1e-6)


def test_sinkhorn_ot_divergence_identical_clouds_is_zero():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(30, 16))
    eps = sinkhorn_epsilon_for_cloud(points, fraction=0.1)
    d = sinkhorn_ot_divergence(points, points, epsilon=eps)
    assert d == pytest.approx(0.0, abs=1e-6)


def test_sinkhorn_ot_divergence_increases_with_perturbation_size():
    rng = np.random.default_rng(0)
    z0 = rng.normal(size=(30, 16))
    eps = sinkhorn_epsilon_for_cloud(z0, fraction=0.1)
    z_small = z0 + rng.normal(scale=0.05, size=(30, 16))
    z_big = z0 + rng.normal(scale=2.0, size=(30, 16))
    d_small = sinkhorn_ot_divergence(z0, z_small, epsilon=eps)
    d_big = sinkhorn_ot_divergence(z0, z_big, epsilon=eps)
    assert 0.0 <= d_small < d_big


def test_sinkhorn_ot_divergence_nonnegative():
    rng = np.random.default_rng(1)
    z0 = rng.normal(size=(30, 16))
    zk = rng.normal(size=(30, 16))
    eps = sinkhorn_epsilon_for_cloud(z0, fraction=0.1)
    d = sinkhorn_ot_divergence(z0, zk, epsilon=eps)
    assert d >= -1e-6
