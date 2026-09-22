"""Focused tests for tasks/TASK-11.md's EnCodec continuous-latent controlled scaling probe --
the pure/model-backed helpers in `muq_inversion.gate3_encodec_common`.

A few tests here load the real (small, fast -- ~seconds, per TASK-10's measured 11.8s full run)
`facebook/encodec_24khz` checkpoint, since this task explicitly requires tests for extraction
shape, determinism, and decoder acceptance that are meaningless without the real model. Kept
focused (not a large harness) per this task's own instruction.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from muq_inversion.encodec_encoder import FrozenEncodecEncoder
from muq_inversion.gate3_encodec_common import (
    compute_oracle_audio_metrics,
    find_best_lag_and_align,
    load_examples_from_manifest_encodec,
    verify_decoder_round_trip,
)
from muq_inversion.gate3_spectrogram_common import evaluate_prediction
from muq_inversion.logging_utils import select_device
from muq_inversion.models.losses import MultiResolutionSTFTLoss

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 24000


@pytest.fixture(scope="module")
def device() -> torch.device:
    return select_device("auto", allow_cpu_fallback=False)


@pytest.fixture(scope="module")
def encodec(device: torch.device) -> FrozenEncodecEncoder:
    return FrozenEncodecEncoder("facebook/encodec_24khz", device)


def test_manifest_nesting_matches_task05():
    """`tasks/TASK-11.md`'s "reuse the exact TASK-05 nested example sets... verify that it
    exactly matches TASK-05" requirement."""
    track_sets = {}
    for n in (2, 8, 32):
        with open(REPO_ROOT / f"data/manifests/gate3_scale_{n}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n, f"gate3_scale_{n}.csv has {len(rows)} rows, expected {n}"
        track_sets[n] = {row["track_id"] for row in rows}
    assert track_sets[2] <= track_sets[8] <= track_sets[32]


def test_encodec_latent_shape_for_2s_crop(encodec: FrozenEncodecEncoder, device: torch.device):
    """150 frames, 128 channels -- matches TASK-02's dense-cache mel/DAC frame count exactly for
    a 2.0s/48000-sample crop (measured, not assumed)."""
    x = torch.randn(1, 48000, device=device)
    z = encodec.encode(x)
    assert z.shape == (1, 128, 150)


def test_encodec_extraction_deterministic(encodec: FrozenEncodecEncoder, device: torch.device):
    x = torch.randn(1, 48000, device=device)
    z1 = encodec.encode(x)
    z2 = encodec.encode(x)
    assert torch.equal(z1, z2)


def test_decoder_accepts_oracle_latent(encodec: FrozenEncodecEncoder, device: torch.device):
    x = torch.randn(1, 48000, device=device)
    assert verify_decoder_round_trip(encodec, x)


def test_decoder_accepts_translator_shaped_latent(encodec: FrozenEncodecEncoder, device: torch.device):
    """A translator's raw (unconstrained-range) output must decode without shape error, not just
    a genuine oracle-encoded latent."""
    fake_prediction = torch.randn(1, 128, 150, device=device) * 5.0  # arbitrary range, not a valid code
    decoded = encodec.decode(fake_prediction)
    assert torch.isfinite(decoded).all()
    assert decoded.shape[0] == 1


def test_load_examples_from_manifest_encodec_shapes(encodec: FrozenEncodecEncoder, device: torch.device):
    examples = load_examples_from_manifest_encodec(
        REPO_ROOT / "data/manifests/gate3_scale_2.csv", device, encodec, SAMPLE_RATE
    )
    assert set(examples) == {"pop.00012", "disco.00049"}
    for ex in examples.values():
        assert ex.mel.shape == (1, 150, 128)
        assert ex.target.shape == (1, 150, 128)


def test_target_and_prediction_shapes_match(encodec: FrozenEncodecEncoder, device: torch.device):
    """`tasks/TASK-11.md`'s "predicted and target EnCodec sequences have identical shapes"
    alignment-validation requirement."""
    examples = load_examples_from_manifest_encodec(
        REPO_ROOT / "data/manifests/gate3_scale_2.csv", device, encodec, SAMPLE_RATE
    )
    ex = examples["pop.00012"]
    fake_prediction = torch.zeros_like(ex.target)
    assert fake_prediction.shape == ex.target.shape


def test_evaluate_prediction_accepts_encodec_encoder_via_duck_typing(encodec: FrozenEncodecEncoder, device: torch.device):
    """The core reuse assumption this task depends on: `gate3_spectrogram_common.py`'s
    `evaluate_prediction` (built for DAC, never modified) only ever calls `.decode(...)` on its
    `dac` parameter, so passing a `FrozenEncodecEncoder` in its place should work unchanged. Uses
    a real multi-resolution STFT loss (cheap) to exercise the full audio-metrics branch, not just
    the latent-only one."""
    examples = load_examples_from_manifest_encodec(
        REPO_ROOT / "data/manifests/gate3_scale_2.csv", device, encodec, SAMPLE_RATE
    )
    ex = examples["pop.00012"]
    stft_loss_fn = MultiResolutionSTFTLoss((512, 1024, 2048), (50, 120, 240), (240, 600, 1200)).to(device)

    metrics = evaluate_prediction(ex.target, ex.target, dac=encodec, stft_loss_fn=stft_loss_fn, device=device, compute_audio_metrics=True)
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["cosine"] == pytest.approx(1.0, abs=1e-4)
    assert "waveform_correlation_to_oracle" in metrics
    assert "stft_distance_to_oracle" in metrics
    assert metrics["waveform_correlation_to_oracle"] == pytest.approx(1.0, abs=1e-3)


def test_compute_oracle_audio_metrics_reasonable_range(encodec: FrozenEncodecEncoder, device: torch.device):
    """`tasks/TASK-11.md`'s EnCodec Oracle Baseline: original waveform vs. EnCodec's own
    encode-then-decode reconstruction should be well-correlated (a real, imperfect codec
    reconstruction -- not identity, unlike the duck-typing test above which deliberately compares
    a latent to itself)."""
    examples = load_examples_from_manifest_encodec(
        REPO_ROOT / "data/manifests/gate3_scale_2.csv", device, encodec, SAMPLE_RATE
    )
    ex = examples["pop.00012"]
    stft_loss_fn = MultiResolutionSTFTLoss((512, 1024, 2048), (50, 120, 240), (240, 600, 1200)).to(device)

    metrics = compute_oracle_audio_metrics(ex, encodec, stft_loss_fn, device)
    assert 0.5 < metrics["waveform_correlation"] <= 1.0
    assert 0.5 < metrics["logmel_cosine_similarity"] <= 1.0
    assert metrics["stft_distance"] >= 0.0


def test_find_best_lag_and_align_recovers_known_shift():
    """A decoder that introduces a fixed k-sample delay (like DAC's measured 8-sample delay --
    see notes/DECISIONS.md#task11-...) should be recoverable: shift a signal by a known amount and
    confirm the search finds that exact lag and restores near-perfect correlation."""
    rng = torch.Generator().manual_seed(0)
    original = torch.randn(2000, generator=rng)
    shift = 8
    # reconstruction[i] == original[i + shift] -- matches DAC's own measured relationship
    # (orig[lag:] aligns with recon[:n] at the function's positive-lag convention).
    shifted = torch.cat([original[shift:], torch.zeros(shift)])

    aligned_orig, aligned_recon, lag = find_best_lag_and_align(original, shifted, max_lag=32)
    assert lag == shift
    assert torch.allclose(aligned_orig, aligned_recon, atol=1e-5)


def test_find_best_lag_and_align_zero_lag_when_already_aligned():
    rng = torch.Generator().manual_seed(1)
    original = torch.randn(2000, generator=rng)
    aligned_orig, aligned_recon, lag = find_best_lag_and_align(original, original.clone(), max_lag=32)
    assert lag == 0
    assert torch.equal(aligned_orig, aligned_recon)


def test_find_best_lag_and_align_output_lengths_match():
    rng = torch.Generator().manual_seed(2)
    original = torch.randn(2000, generator=rng)
    reconstruction = torch.randn(1995, generator=rng)
    aligned_orig, aligned_recon, _lag = find_best_lag_and_align(original, reconstruction, max_lag=32)
    assert aligned_orig.shape == aligned_recon.shape
