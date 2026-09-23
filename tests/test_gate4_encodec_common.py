"""Tests for `tasks/TASK-13.md`'s Gate 4 EnCodec-revalidation helpers (`gate4_encodec_common.py`).

Per this task's explicit "verify control construction with tests rather than assuming it is
correct" requirement -- the four `build_conditioning` modes are tested against synthetic tensors
(no model needed), and manifest loading/frame-ratio verification is tested against the real
(small, fast) MuQ + EnCodec checkpoints, mirroring `test_gate3_encodec_common.py`'s convention.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from muq_inversion.encodec_encoder import FrozenEncodecEncoder
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.gate4_encodec_common import (
    CONDITIONING_MODES,
    Gate4CropSpec,
    Gate4Example,
    build_conditioning,
    load_gate4_examples,
    load_gate4_manifest,
)
from muq_inversion.logging_utils import select_device

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/gate4_crops.csv"
SAMPLE_RATE = 24000


def _make_spec(track_id: str, split: str = "train", crop_index: int = 0) -> Gate4CropSpec:
    return Gate4CropSpec(
        experiment="B", split=split, track_id=track_id, genre="pop",
        path=REPO_ROOT / "data/raw/gtzan/genres/pop/pop.00001.wav",
        checksum_sha256="deadbeef", crop_index=crop_index,
        crop_start_seconds=0.0, crop_duration_seconds=2.0,
    )


def _make_example(track_id: str, num_frames: int = 10, channels: int = 1024, seed: int = 0) -> Gate4Example:
    gen = torch.Generator().manual_seed(seed)
    muq = torch.randn(1, num_frames, channels, generator=gen)
    target = torch.randn(1, num_frames * 3, 128, generator=gen)
    waveform = torch.randn(1, num_frames * 3 * 320, generator=gen)
    return Gate4Example(spec=_make_spec(track_id), muq=muq, target=target, waveform=waveform)


# ---- Manifest loading (pure, no model) ----


def test_load_gate4_manifest_experiment_b_splits():
    specs = load_gate4_manifest(MANIFEST_PATH, "B")
    by_split: dict[str, set[str]] = {}
    for spec in specs:
        by_split.setdefault(spec.split, set()).add(spec.track_id)
    assert by_split["train"] == {
        "classical.00001", "classical.00002", "disco.00001", "disco.00002",
        "pop.00001", "pop.00002", "rock.00001", "rock.00002",
    }
    assert by_split["val"] == {"disco.00003", "pop.00003"}
    assert by_split["test"] == {"classical.00003", "rock.00003"}
    # No track appears in more than one split -- track-level generalization by construction.
    all_tracks = [s.track_id for s in specs]
    track_to_splits: dict[str, set[str]] = {}
    for spec in specs:
        track_to_splits.setdefault(spec.track_id, set()).add(spec.split)
    for track_id, splits_seen in track_to_splits.items():
        assert len(splits_seen) == 1, f"{track_id} spans {splits_seen}"
    assert len(all_tracks) == 20


def test_load_gate4_manifest_unknown_experiment_returns_empty():
    assert load_gate4_manifest(MANIFEST_PATH, "Z") == []


# ---- build_conditioning: pure, synthetic-tensor tests ----


def test_conditioning_modes_tuple_matches_task13():
    assert CONDITIONING_MODES == ("correct", "shuffled_across", "shuffled_time", "constant_zero")


def test_correct_conditioning_is_unmodified_repeat_interleave():
    ex = _make_example("pop.00001")
    result = build_conditioning("correct", ex, [], frame_ratio=3, rng=random.Random(0))
    assert torch.equal(result, ex.muq.repeat_interleave(3, dim=1))
    assert result.shape[1] == ex.muq.shape[1] * 3


def test_shuffled_across_uses_a_different_tracks_features():
    target_ex = _make_example("pop.00001", seed=1)
    other_ex = _make_example("rock.00001", seed=2)
    result = build_conditioning(
        "shuffled_across", target_ex, [target_ex, other_ex], frame_ratio=3, rng=random.Random(0)
    )
    assert torch.equal(result, other_ex.muq.repeat_interleave(3, dim=1))
    assert not torch.equal(result, target_ex.muq.repeat_interleave(3, dim=1))


def test_shuffled_across_never_picks_the_same_track_even_with_duplicates():
    """`other_examples` may contain a second crop from the same track (as Experiment B's train
    split does, 2 crops/track) -- shuffled_across must never resolve to the same track_id."""
    target_ex = _make_example("pop.00001", seed=1)
    same_track_other_crop = _make_example("pop.00001", seed=3)
    diff_track = _make_example("rock.00001", seed=2)
    rng = random.Random(42)
    for _ in range(20):
        result = build_conditioning(
            "shuffled_across", target_ex, [target_ex, same_track_other_crop, diff_track],
            frame_ratio=3, rng=rng,
        )
        assert torch.equal(result, diff_track.muq.repeat_interleave(3, dim=1))


def test_shuffled_across_raises_when_no_other_track_available():
    target_ex = _make_example("pop.00001")
    with pytest.raises(ValueError, match="no other-track candidates"):
        build_conditioning("shuffled_across", target_ex, [target_ex], frame_ratio=3, rng=random.Random(0))


def test_shuffled_time_permutes_frames_not_channels():
    ex = _make_example("pop.00001", num_frames=12)
    result = build_conditioning("shuffled_time", ex, [], frame_ratio=3, rng=random.Random(0))
    repeated_correct = ex.muq.repeat_interleave(3, dim=1)
    assert result.shape == repeated_correct.shape
    assert not torch.equal(result, repeated_correct)
    # Same multiset of frame vectors, just reordered -- not a content change.
    sorted_result = sorted(result[0, ::3].tolist())
    sorted_correct = sorted(ex.muq[0].tolist())
    assert sorted_result == sorted_correct


def test_shuffled_time_is_a_true_permutation_not_a_fixed_shift():
    """Distinguishes this control from the older `shifted_time_control` (a fixed circular roll,
    `gate4_common.py`) -- `tasks/TASK-13.md` requires frames "permuted in time", and a genuine
    random permutation should not equal every possible circular shift of the same sequence."""
    ex = _make_example("pop.00001", num_frames=20, seed=7)
    result = build_conditioning("shuffled_time", ex, [], frame_ratio=1, rng=random.Random(0))
    matches_some_shift = any(
        torch.equal(result, torch.roll(ex.muq, shifts=k, dims=1)) for k in range(20)
    )
    assert not matches_some_shift


def test_shuffled_time_is_deterministic_given_the_same_rng_sequence():
    ex = _make_example("pop.00001", num_frames=12)
    result_a = build_conditioning("shuffled_time", ex, [], frame_ratio=3, rng=random.Random(123))
    result_b = build_conditioning("shuffled_time", ex, [], frame_ratio=3, rng=random.Random(123))
    assert torch.equal(result_a, result_b)


def test_constant_zero_is_all_zero_and_correct_shape():
    ex = _make_example("pop.00001", num_frames=8)
    result = build_conditioning("constant_zero", ex, [], frame_ratio=3, rng=random.Random(0))
    assert torch.equal(result, torch.zeros(1, 24, 1024))


def test_unknown_mode_raises():
    ex = _make_example("pop.00001")
    with pytest.raises(ValueError, match="unknown conditioning mode"):
        build_conditioning("bogus_mode", ex, [], frame_ratio=3, rng=random.Random(0))


# ---- Real-model tests: manifest -> MuQ/EnCodec extraction, frame-ratio verification ----


@pytest.fixture(scope="module")
def device() -> torch.device:
    return select_device("auto", allow_cpu_fallback=False)


@pytest.fixture(scope="module")
def muq(device: torch.device) -> FrozenMuQEncoder:
    return FrozenMuQEncoder("OpenMuQ/MuQ-large-msd-iter", device)


@pytest.fixture(scope="module")
def encodec(device: torch.device) -> FrozenEncodecEncoder:
    return FrozenEncodecEncoder("facebook/encodec_24khz", device)


def test_load_gate4_examples_shapes_and_frame_ratio(muq, encodec, device):
    examples, frame_ratio = load_gate4_examples(MANIFEST_PATH, "B", muq, encodec, SAMPLE_RATE, device)
    assert len(examples) == 20
    assert frame_ratio == 3  # MuQ 25Hz -> EnCodec 75Hz, matches every prior MuQ-to-codec experiment
    for ex in examples.values():
        assert ex.muq.shape[0] == 1
        assert ex.muq.shape[2] == 1024
        assert ex.target.shape[0] == 1
        assert ex.target.shape[2] == 128
        assert ex.target.shape[1] == ex.muq.shape[1] * frame_ratio


def test_load_gate4_examples_targets_are_not_inference_tensors(muq, encodec, device):
    """Regression test for TASK-11's inference-tensor autograd bug -- targets must be usable in
    a real backward pass."""
    examples, _ = load_gate4_examples(MANIFEST_PATH, "B", muq, encodec, SAMPLE_RATE, device)
    ex = next(iter(examples.values()))
    pred = ex.target.clone().requires_grad_(True)
    loss = torch.nn.functional.mse_loss(pred, ex.target)
    loss.backward()  # must not raise "Inference tensors cannot be saved for backward"
    assert pred.grad is not None
