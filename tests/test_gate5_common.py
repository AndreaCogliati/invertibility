"""Tests for `tasks/TASK-14.md`'s Gate 5 full-GTZAN pilot helpers (`gate5_common.py`).

`TrainCropSampler` and `paired_delta_stats` are pure/synthetic-data tests (no model needed);
`load_train_tracks` is tested against the real committed manifest.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from muq_inversion.data import TrackRecord
from muq_inversion.encodec_encoder import FrozenEncodecEncoder
from muq_inversion.gate5_common import (
    TrainCropSampler,
    compute_held_out_audio_metrics,
    load_gate5_config,
    load_train_crop_waveform,
    load_train_tracks,
    oracle_relative_recovery,
    paired_delta_stats,
    write_limited_heldout_manifest,
)
from muq_inversion.logging_utils import select_device
from muq_inversion.models.losses import MultiResolutionSTFTLoss

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/gtzan_manifest_smoke.csv"
FULL_CONFIG_PATH = REPO_ROOT / "configs/gate5_full_gtzan.yaml"
SMOKE_CONFIG_PATH = REPO_ROOT / "configs/gate5_smoke.yaml"


def _make_track(track_id: str, duration: float = 30.0, split: str = "train") -> TrackRecord:
    return TrackRecord(
        track_id=track_id, genre="pop", relative_path=f"genres/pop/{track_id}.wav",
        checksum=f"checksum-{track_id}", native_sample_rate=22050, native_num_frames=661500,
        duration_seconds=duration, corrupted=False, split=split,
    )


# ---- TrainCropSampler ----


def test_sampler_visits_only_provided_tracks():
    tracks = [_make_track("a"), _make_track("b"), _make_track("c")]
    sampler = TrainCropSampler(tracks=tracks, crop_duration_seconds=2.0, rng=random.Random(0))
    for _ in range(200):
        draw = sampler.draw()
        assert draw.track_id in {"a", "b", "c"}
        assert 0.0 <= draw.crop_start_seconds <= 30.0 - 2.0


def test_sampler_crop_start_never_exceeds_track_bounds():
    tracks = [_make_track("short", duration=3.0)]
    sampler = TrainCropSampler(tracks=tracks, crop_duration_seconds=2.0, rng=random.Random(1))
    for _ in range(100):
        draw = sampler.draw()
        assert 0.0 <= draw.crop_start_seconds <= 1.0 + 1e-9  # max_start = 3.0 - 2.0 = 1.0


def test_sampler_rejects_tracks_shorter_than_crop_duration():
    tracks = [_make_track("tiny", duration=1.0)]
    with pytest.raises(ValueError, match="shorter than crop_duration_seconds"):
        TrainCropSampler(tracks=tracks, crop_duration_seconds=2.0, rng=random.Random(0))


def test_sampler_exposure_counts_accumulate():
    tracks = [_make_track("a"), _make_track("b")]
    sampler = TrainCropSampler(tracks=tracks, crop_duration_seconds=2.0, rng=random.Random(0))
    for _ in range(50):
        sampler.draw()
    assert sum(sampler.exposure_counts.values()) == 50
    assert sampler.num_draws == 50
    summary = sampler.exposure_summary()
    assert summary["tracks_total"] == 2
    assert summary["tracks_visited"] <= 2


def test_sampler_broadly_visits_many_tracks_not_a_small_subset():
    """The core requirement this sampler exists for: with enough draws over many tracks, most
    tracks should be visited at least once -- not concentrated on a handful."""
    tracks = [_make_track(f"t{i}") for i in range(100)]
    sampler = TrainCropSampler(tracks=tracks, crop_duration_seconds=2.0, rng=random.Random(0))
    for _ in range(5000):
        sampler.draw()
    summary = sampler.exposure_summary()
    assert summary["tracks_visited"] >= 90  # expect near-total coverage at 50x draws/track


def test_sampler_state_dict_round_trip_reproduces_future_draws():
    tracks = [_make_track(f"t{i}") for i in range(10)]
    sampler_a = TrainCropSampler(tracks=tracks, crop_duration_seconds=2.0, rng=random.Random(42))
    for _ in range(30):
        sampler_a.draw()
    state = sampler_a.state_dict()
    future_a = [sampler_a.draw() for _ in range(20)]

    sampler_b = TrainCropSampler(tracks=tracks, crop_duration_seconds=2.0, rng=random.Random(999))
    sampler_b.load_state_dict(state)
    future_b = [sampler_b.draw() for _ in range(20)]

    assert [d.track_id for d in future_a] == [d.track_id for d in future_b]
    assert [d.crop_start_seconds for d in future_a] == [d.crop_start_seconds for d in future_b]
    assert sampler_b.exposure_counts == sampler_a.exposure_counts
    assert sampler_b.num_draws == sampler_a.num_draws


def test_sampler_never_needs_val_or_test_tracks_to_be_passed():
    """Documents the "never samples validation or test tracks" requirement structurally: the
    sampler only ever draws from whatever `tracks` list it's constructed with, so passing only
    train-split tracks (as `load_train_tracks` guarantees) is sufficient -- there is no code path
    that could reach a val/test track once excluded at construction time."""
    tracks = [_make_track("only_train_track")]
    sampler = TrainCropSampler(tracks=tracks, crop_duration_seconds=2.0, rng=random.Random(0))
    for _ in range(20):
        assert sampler.draw().track_id == "only_train_track"


# ---- load_train_tracks (real manifest) ----


def test_load_train_tracks_matches_committed_split_counts():
    tracks = load_train_tracks(MANIFEST_PATH)
    assert len(tracks) == 792  # 793 train rows - 1 known-corrupted (jazz.00054, per Gate 1)
    assert all(t.split == "train" for t in tracks)
    assert all(not t.corrupted for t in tracks)


def test_load_train_tracks_excludes_val_and_test():
    tracks = load_train_tracks(MANIFEST_PATH)
    track_ids = {t.track_id for t in tracks}
    assert "jazz.00054" not in track_ids  # the known-corrupted train-split track


# ---- load_train_crop_waveform (real audio) ----


def test_load_train_crop_waveform_shape_and_finite():
    tracks = load_train_tracks(MANIFEST_PATH)
    sampler = TrainCropSampler(tracks=tracks[:5], crop_duration_seconds=2.0, rng=random.Random(0))
    draw = sampler.draw()
    waveform = load_train_crop_waveform(draw, REPO_ROOT / "data/raw/gtzan", sample_rate=24000)
    assert waveform.shape == (1, 48000)
    assert torch.isfinite(waveform).all()


# ---- write_limited_heldout_manifest ----


def test_write_limited_heldout_manifest_bounds_val_and_test(tmp_path):
    heldout_path = REPO_ROOT / "data/manifests/gate5_heldout_crops.csv"
    dest = tmp_path / "limited.csv"
    write_limited_heldout_manifest(heldout_path, dest, crop_limit=3)

    import csv
    with open(dest, newline="") as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    assert len(val_rows) == 3
    assert len(test_rows) == 3
    assert all(r["experiment"] == "gate5" for r in rows)


def test_write_limited_heldout_manifest_is_a_deterministic_prefix(tmp_path):
    heldout_path = REPO_ROOT / "data/manifests/gate5_heldout_crops.csv"
    dest_a = tmp_path / "a.csv"
    dest_b = tmp_path / "b.csv"
    write_limited_heldout_manifest(heldout_path, dest_a, crop_limit=2)
    write_limited_heldout_manifest(heldout_path, dest_b, crop_limit=2)
    assert dest_a.read_text() == dest_b.read_text()


# ---- oracle_relative_recovery ----


def test_oracle_relative_recovery_basic_ratio():
    assert oracle_relative_recovery(0.75, 1.0) == pytest.approx(0.75)
    assert oracle_relative_recovery(0.723, 0.914) == pytest.approx(0.723 / 0.914)


def test_oracle_relative_recovery_zero_oracle_is_nan():
    import math
    assert math.isnan(oracle_relative_recovery(0.5, 0.0))


# ---- paired_delta_stats ----


def test_paired_delta_stats_requires_matched_lengths():
    with pytest.raises(ValueError, match="matched pairs"):
        paired_delta_stats([1.0, 2.0], [1.0])


def test_paired_delta_stats_requires_nonempty():
    with pytest.raises(ValueError, match="at least one paired example"):
        paired_delta_stats([], [])


def test_paired_delta_stats_all_positive_deltas():
    correct = [0.9, 0.8, 0.95, 0.85]
    control = [0.1, 0.05, 0.2, 0.0]
    result = paired_delta_stats(correct, control, seed=0, num_bootstrap=500)
    assert result["n"] == 4
    assert result["proportion_positive"] == 1.0
    assert result["mean_delta"] > 0.5
    assert result["bootstrap_ci_low"] > 0  # entirely positive deltas -> positive CI


def test_paired_delta_stats_zero_deltas_when_identical():
    values = [0.5, 0.6, 0.7]
    result = paired_delta_stats(values, values, seed=0, num_bootstrap=200)
    assert result["mean_delta"] == pytest.approx(0.0)
    assert result["median_delta"] == pytest.approx(0.0)
    assert result["proportion_positive"] == 0.0


def test_paired_delta_stats_deterministic_given_same_seed():
    correct = [0.9, 0.3, 0.6, 0.5, 0.1]
    control = [0.2, 0.4, 0.1, 0.5, 0.3]
    r1 = paired_delta_stats(correct, control, seed=7, num_bootstrap=300)
    r2 = paired_delta_stats(correct, control, seed=7, num_bootstrap=300)
    assert r1 == r2


# ---- Gate5Config / load_gate5_config ----


def test_load_full_gtzan_config():
    config = load_gate5_config(FULL_CONFIG_PATH)
    assert config.task == "TASK-14"
    assert config.run_label == "gate5_full_gtzan"
    assert config.steps == 300000
    assert config.adam_betas == (0.9, 0.999)
    assert config.train_track_limit is None
    assert config.heldout_crop_limit is None
    assert "task14" in config.wandb_tags


def test_load_smoke_config():
    config = load_gate5_config(SMOKE_CONFIG_PATH)
    assert config.run_label == "gate5_smoke"
    assert config.steps == 20
    assert config.train_track_limit == 4
    assert config.heldout_crop_limit == 4
    # Smoke and full runs must never collide on output paths.
    full_config = load_gate5_config(FULL_CONFIG_PATH)
    assert config.run_dir_name != full_config.run_dir_name
    assert config.artifact_dir_name != full_config.artifact_dir_name


def test_load_gate5_config_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(FULL_CONFIG_PATH.read_text() + "\nbogus_key: 1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_gate5_config(bad)


# ---- compute_held_out_audio_metrics (real EnCodec model) ----


@pytest.fixture(scope="module")
def device() -> torch.device:
    return select_device("auto", allow_cpu_fallback=False)


@pytest.fixture(scope="module")
def encodec(device: torch.device) -> FrozenEncodecEncoder:
    return FrozenEncodecEncoder("facebook/encodec_24khz", device)


@pytest.fixture(scope="module")
def stft_loss_fn(device: torch.device):
    return MultiResolutionSTFTLoss((512, 1024, 2048), (50, 120, 240), (240, 600, 1200)).to(device)


def test_compute_held_out_audio_metrics_identical_prediction_and_oracle(encodec, device, stft_loss_fn):
    """When the "translator" prediction equals the oracle exactly, translator_to_oracle
    correlation should be ~1.0 and oracle-relative recovery should be ~1.0."""
    gen = torch.Generator().manual_seed(0)
    original = torch.randn(48000, generator=gen)
    latent = encodec.encode(original.unsqueeze(0).to(device))
    oracle_audio = encodec.decode(latent).squeeze(0)

    metrics = compute_held_out_audio_metrics(original, oracle_audio, oracle_audio.clone(), stft_loss_fn, device)
    assert metrics["translator_to_oracle_correlation"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["oracle_relative_waveform_recovery"] == pytest.approx(1.0, abs=1e-2)
    assert metrics["lag_applied_original_vs_translator"] == 0


def test_compute_held_out_audio_metrics_uncorrelated_prediction(encodec, device, stft_loss_fn):
    # A sine tone, not white noise -- EnCodec (a music codec) reconstructs tonal content far
    # better than white noise, giving a meaningful "real reconstruction" oracle correlation here.
    t = torch.arange(48000, dtype=torch.float32) / 24000.0
    original = 0.5 * torch.sin(2 * torch.pi * 220.0 * t)
    latent = encodec.encode(original.unsqueeze(0).to(device))
    oracle_audio = encodec.decode(latent).squeeze(0)
    unrelated_prediction = torch.randn(48000, generator=torch.Generator().manual_seed(99))

    metrics = compute_held_out_audio_metrics(original, oracle_audio, unrelated_prediction, stft_loss_fn, device)
    assert abs(metrics["original_to_translator_correlation"]) < 0.2
    assert metrics["original_to_oracle_correlation"] > 0.5  # the oracle itself is a real reconstruction
