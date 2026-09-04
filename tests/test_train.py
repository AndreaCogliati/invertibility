import random

import pytest
import torch

from muq_inversion.config import DataConfig, TrainingConfig
from muq_inversion.data import TrackRecord
from muq_inversion.train import (
    OverfitClip,
    fixed_validation_crop,
    sample_training_crop,
    select_overfit_tracks,
)


class _StubConfig:
    def __init__(self, training: TrainingConfig, data: DataConfig):
        self.training = training
        self.data = data


def _make_record(track_id: str, split: str = "train", corrupted: bool = False) -> TrackRecord:
    return TrackRecord(
        track_id=track_id,
        genre="blues",
        relative_path=f"blues/{track_id}.wav",
        checksum=f"checksum-{track_id}",
        native_sample_rate=22050,
        native_num_frames=661794,
        duration_seconds=30.0,
        corrupted=corrupted,
        split=split,
    )


def test_select_overfit_tracks_deterministic_default():
    records = [_make_record(f"blues.{i:05d}") for i in range(10)]
    config = _StubConfig(TrainingConfig(num_overfit_tracks=2), DataConfig())
    selected_a = select_overfit_tracks(config, records)
    selected_b = select_overfit_tracks(config, records)
    assert len(selected_a) == 2
    assert [r.track_id for r in selected_a] == [r.track_id for r in selected_b]


def test_select_overfit_tracks_excludes_corrupted_and_non_train():
    records = [
        _make_record("blues.00000", split="train", corrupted=True),
        _make_record("blues.00001", split="val"),
        _make_record("blues.00002", split="train"),
    ]
    config = _StubConfig(TrainingConfig(num_overfit_tracks=2), DataConfig())
    selected = select_overfit_tracks(config, records)
    assert [r.track_id for r in selected] == ["blues.00002"]


def test_select_overfit_tracks_explicit_ids():
    records = [_make_record(f"blues.{i:05d}") for i in range(5)]
    config = _StubConfig(TrainingConfig(overfit_track_ids=("blues.00003", "blues.00001")), DataConfig())
    selected = select_overfit_tracks(config, records)
    assert [r.track_id for r in selected] == ["blues.00003", "blues.00001"]


def test_select_overfit_tracks_raises_on_missing_explicit_id():
    records = [_make_record("blues.00000")]
    config = _StubConfig(TrainingConfig(overfit_track_ids=("does.not.exist",)), DataConfig())
    with pytest.raises(ValueError):
        select_overfit_tracks(config, records)


def test_sample_training_crop_has_exact_length():
    clip = OverfitClip(track_id="x", waveform=torch.arange(1000, dtype=torch.float32))
    rng = random.Random(0)
    crop = sample_training_crop(clip, crop_samples=100, rng=rng)
    assert crop.shape == (100,)


def test_sample_training_crop_rejects_too_short_clip():
    clip = OverfitClip(track_id="x", waveform=torch.zeros(50))
    with pytest.raises(ValueError):
        sample_training_crop(clip, crop_samples=100, rng=random.Random(0))


def test_fixed_validation_crop_is_deterministic_prefix():
    waveform = torch.arange(1000, dtype=torch.float32)
    clip = OverfitClip(track_id="x", waveform=waveform)
    crop = fixed_validation_crop(clip, crop_samples=100)
    assert torch.equal(crop, waveform[:100])
