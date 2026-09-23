"""Tests for tasks/TASK-04.md's spectrogram Gate-3 memorization control
(`src/muq_inversion/gate3_spectrogram_common.py`, `scripts/train_spectrogram_gate3_wandb.py`),
per this task's explicit test-coverage list: W&B disabled mode, metric logging payload
construction, deterministic two-example loading, checkpoint save/reload consistency, per-example
metric separation, input-target alignment assertions.

Tests that need the 10.36GB dense cache are skipped when it isn't present in the current
environment (mirrors `test_dense_manifest.py`'s pattern).
"""

from __future__ import annotations

import csv

import pytest
import torch

from muq_inversion.gate3_spectrogram_common import (
    DENSE_CACHE_PATH,
    LABELS,
    MANIFEST_PATH,
    TRACKS,
    build_memorization_log_payload,
    count_nonfinite,
    evaluate_prediction,
    load_examples,
    three_way_grad_norms,
)
from muq_inversion.models.spectrogram_translator import SpectrogramToDacTranslator
from muq_inversion.spectrogram_encoder import N_MELS
from muq_inversion.wandb_logging import init_run

requires_dense_cache = pytest.mark.skipif(
    not DENSE_CACHE_PATH.exists(), reason="data/cache/dense/mel_dac_cache.pt not present -- run scripts/precompute_dense_cache.py"
)


# --- W&B disabled mode ---


def test_init_run_disabled_mode_does_not_require_network():
    """`mode='disabled'` must work with no credentials and no network access."""
    environment = {"git_commit": "deadbeef", "hostname": "test-host", "machine": "test"}
    run = init_run(config={"foo": "bar"}, environment=environment, mode="disabled", group="spectrogram-gate3", job_type="test")
    assert run is not None
    run.log({"x": 1}, step=0)  # must not raise
    run.finish()


# --- metric logging payload construction ---


def test_build_memorization_log_payload_keys():
    per_example = {
        "pop.00012": {"mse": 0.1, "cosine": 0.9},
        "disco.00049": {"mse": 0.2, "cosine": 0.8},
    }
    mean_metrics = {"mse": 0.15, "cosine": 0.85}
    payload = build_memorization_log_payload(per_example, mean_metrics, TRACKS)

    assert payload["memorization/example_0/mse"] == 0.1
    assert payload["memorization/example_0/cosine"] == 0.9
    assert payload["memorization/example_1/mse"] == 0.2
    assert payload["memorization/example_1/cosine"] == 0.8
    assert payload["memorization/mean_mse"] == 0.15
    assert payload["memorization/mean_cosine"] == 0.85


def test_build_memorization_log_payload_no_cross_contamination():
    """Example 0's metrics must never leak into example 1's keys or vice versa -- the exact
    per-example-metric-separation property this task asks to be tested."""
    per_example = {
        "pop.00012": {"mse": 111.0},
        "disco.00049": {"mse": 222.0},
    }
    payload = build_memorization_log_payload(per_example, {"mse": 166.5}, TRACKS)
    assert payload["memorization/example_0/mse"] == 111.0
    assert payload["memorization/example_1/mse"] == 222.0
    assert payload["memorization/example_0/mse"] != payload["memorization/example_1/mse"]


# --- per-example metric separation (evaluate_prediction) ---


def test_evaluate_prediction_distinguishes_different_examples():
    """Two structurally different (pred, target) pairs must produce different metrics -- proof
    `evaluate_prediction` doesn't silently average or conflate examples."""
    torch.manual_seed(0)
    target_a = torch.randn(1, 50, 1024) * 5.0
    pred_a = target_a + torch.randn(1, 50, 1024) * 0.1  # close match
    target_b = torch.randn(1, 50, 1024) * 5.0
    pred_b = torch.randn(1, 50, 1024) * 5.0  # unrelated

    metrics_a = evaluate_prediction(pred_a, target_a, dac=None, stft_loss_fn=None, device=torch.device("cpu"), compute_audio_metrics=False)
    metrics_b = evaluate_prediction(pred_b, target_b, dac=None, stft_loss_fn=None, device=torch.device("cpu"), compute_audio_metrics=False)

    assert metrics_a["mse"] < metrics_b["mse"]
    assert metrics_a["cosine"] > metrics_b["cosine"]


def test_evaluate_prediction_perfect_match_gives_near_zero_mse_and_cosine_one():
    target = torch.randn(1, 30, 1024)
    metrics = evaluate_prediction(target, target, dac=None, stft_loss_fn=None, device=torch.device("cpu"), compute_audio_metrics=False)
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["norm_ratio"] == pytest.approx(1.0, abs=1e-4)


def test_evaluate_prediction_skips_audio_metrics_when_not_requested():
    target = torch.randn(1, 20, 1024)
    metrics = evaluate_prediction(target, target, dac=None, stft_loss_fn=None, device=torch.device("cpu"), compute_audio_metrics=False)
    assert "waveform_correlation_to_oracle" not in metrics
    assert "stft_distance_to_oracle" not in metrics


# --- gradient diagnostics (mel_projection / translator / output_layer split) ---


def test_three_way_grad_norms_all_nonzero_after_backward():
    model = SpectrogramToDacTranslator(input_channels=N_MELS)
    x = torch.randn(1, 30, N_MELS)
    model(x).sum().backward()
    norms = three_way_grad_norms(model)
    assert set(norms.keys()) == {"mel_projection_l2_norm", "translator_l2_norm", "output_layer_l2_norm"}
    assert all(v > 0 for v in norms.values())


def test_three_way_grad_norms_zero_before_backward():
    model = SpectrogramToDacTranslator(input_channels=N_MELS)
    norms = three_way_grad_norms(model)
    assert all(v == 0.0 for v in norms.values())


def test_count_nonfinite_detects_nan_and_inf():
    clean = torch.tensor([1.0, 2.0, 3.0])
    dirty = torch.tensor([1.0, float("nan"), float("inf")])
    assert count_nonfinite([clean]) == 0
    assert count_nonfinite([dirty]) == 2
    assert count_nonfinite([clean, dirty]) == 2


# --- checkpoint save/reload consistency ---


def test_checkpoint_save_reload_reproduces_predictions(tmp_path):
    model = SpectrogramToDacTranslator(input_channels=N_MELS)
    model.eval()
    x = torch.randn(1, 40, N_MELS)
    with torch.no_grad():
        pred_before = model(x).clone()

    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save({"step": 123, "model_state_dict": model.state_dict()}, ckpt_path)

    reloaded = SpectrogramToDacTranslator(input_channels=N_MELS)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(state["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        pred_after = reloaded(x)

    assert torch.allclose(pred_before, pred_after, atol=1e-7)
    assert state["step"] == 123


# --- deterministic two-example loading / input-target alignment ---


def test_manifest_contains_exactly_the_two_gate3_tracks():
    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    track_ids = {row["track_id"] for row in rows}
    assert track_ids == set(TRACKS)
    assert len(rows) == 2


def test_manifest_crop_matches_labels_dict():
    with open(MANIFEST_PATH, newline="") as f:
        rows = {row["track_id"]: row for row in csv.DictReader(f)}
    for track_id in TRACKS:
        row = rows[track_id]
        expected_label = f"{row['split']}_{track_id}_t{float(row['crop_start_seconds']):.1f}s"
        assert LABELS[track_id] == expected_label


@requires_dense_cache
def test_load_examples_is_deterministic():
    examples_a = load_examples(torch.device("cpu"))
    examples_b = load_examples(torch.device("cpu"))
    for track_id in TRACKS:
        assert torch.equal(examples_a[track_id].mel, examples_b[track_id].mel)
        assert torch.equal(examples_a[track_id].dac, examples_b[track_id].dac)


@requires_dense_cache
def test_load_examples_input_target_frame_alignment():
    examples = load_examples(torch.device("cpu"))
    for track_id in TRACKS:
        ex = examples[track_id]
        assert ex.mel.shape[1] == ex.dac.shape[1] == 150
        assert ex.mel.shape[-1] == N_MELS
        assert ex.dac.shape[-1] == 1024
