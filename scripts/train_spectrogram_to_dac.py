"""Positive-control experiment (`tasks/TASK-01.md`): does the existing MuQ-to-DAC training
pipeline -- architecture, loss, optimizer, dataset, checkpoint selection -- successfully learn a
mapping when given a representation known to retain far more reconstruction-relevant information
than MuQ?

    log-mel spectrogram -> input projection -> existing TemporalConvNet -> DAC continuous latent
        -> frozen DAC decoder -> reconstructed audio

This is NOT an attempt to improve MuQ inversion. Everything that can be held identical to
`experiment/muq-to-dac-more-data` (the current best MuQ-to-DAC model, 60000 steps, best
checkpoint step 50000 -- `artifacts/more_data/report.json`) IS held identical: manifest, split,
crop selection, `TemporalConvNet`, optimizer, learning rate, loss (MSE + 0.1*cosine), step budget,
checkpoint-selection rule, evaluation protocol (correct / shuffled-across / time-shifted
conditioning), DAC latent target and decoder. The only two differences, both forced by using a
different input representation, are:

1. The input itself: log-mel spectrogram (`spectrogram_encoder.py`, 128 mel bins, aligned to
   DAC's 75Hz grid -- no repeat-interleave upsampling step needed, unlike MuQ's 25Hz features) in
   place of MuQ's frozen `last_hidden_state`.
2. A single learned linear input projection (128 -> 1024) prepended to the otherwise-unmodified
   `TemporalConvNet`, required because the spectrogram's channel count does not coincidentally
   equal DAC's the way MuQ's did (see `models/spectrogram_translator.py`).

Required diagnostics beyond the more_data script's own metrics (this task's explicit ask):
latent norm ratio, latent variance ratio, temporal variance, and lag-1 autocorrelation, computed
predicted-vs-oracle on held-out (val, test) splits (`latent_diagnostics.py`) -- to check whether
norm/variance collapse persists when the input representation is swapped.

Findings and full write-up: notes/DECISIONS.md#spectrogram-positive-control, notes/SCRATCHPAD.md.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

from muq_inversion.dac_encoder import FrozenDacEncoder
from muq_inversion.gate4_common import (
    CropSpec,
    load_crop_waveform,
    shifted_time_control,
    waveform_correlation,
)
from muq_inversion.latent_diagnostics import compute_latent_stats, ratio
from muq_inversion.logging_utils import DeviceSelectionError, select_device
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.models.spectrogram_translator import SpectrogramToDacTranslator
from muq_inversion.spectrogram_encoder import N_MELS

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/more_data_crops.csv"  # reused unchanged
DAC_CACHE_PATH = REPO_ROOT / "data/cache/more_data/muq_dac_cache.pt"  # reused unchanged, read-only
MEL_CACHE_PATH = REPO_ROOT / "data/cache/spectrogram_control/mel_cache.pt"
MORE_DATA_REPORT_PATH = REPO_ROOT / "artifacts/more_data/report.json"  # comparison baseline

SAMPLE_RATE = 24000
DAC_MODEL_ID = "descript/dac_24khz"  # decoding only, not encoding

# Unchanged from experiment/muq-to-dac-more-data.
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
COSINE_LOSS_WEIGHT = 0.1
STEPS = 60000
LOG_EVERY = 5000
VAL_EVERY = 1000
SEED = 0
TIME_SHIFT_FRAMES = 10

STFT_FFT_SIZES = (512, 1024, 2048)
STFT_HOP_SIZES = (50, 120, 240)
STFT_WIN_LENGTHS = (240, 600, 1200)

RUN_ID = "spectrogram_control"
LISTENING_VAL_TRACKS = 3
LISTENING_TEST_TRACKS = 3


def _save_atomic(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _write_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().numpy().astype(np.float32), sample_rate)


def load_manifest() -> list[CropSpec]:
    specs = []
    with open(MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            specs.append(
                CropSpec(
                    experiment="spectrogram_control",
                    split=row["split"],
                    track_id=row["track_id"],
                    genre=row["genre"],
                    path=REPO_ROOT / row["path"],
                    checksum_sha256=row["checksum_sha256"],
                    crop_index=int(row["crop_index"]),
                    crop_start_seconds=float(row["crop_start_seconds"]),
                    crop_duration_seconds=float(row["crop_duration_seconds"]),
                )
            )
    return specs


def run_tiny_subset_sanity_check(mel_features: dict, dac_targets: dict, device: torch.device) -> None:
    """Mirrors every prior MuQ-to-DAC experiment's pre-flight check: a fresh model overfitting
    ONE cached crop must converge fast (mse ratio last10/first10 <= 0.5 over 300 steps)."""
    print("\n--- Pre-flight: tiny-subset sanity check (must reproduce prior experiments' overfit pace) ---")
    label = next(iter(mel_features))
    mel_feat, dac_target = mel_features[label], dac_targets[label]

    torch.manual_seed(SEED)
    model = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()
    history: list[float] = []
    for _ in range(300):
        predicted = model(mel_feat)
        mse = mse_loss_fn(predicted, dac_target)
        cos_sim = nn.functional.cosine_similarity(predicted, dac_target, dim=-1).mean()
        loss = mse + COSINE_LOSS_WEIGHT * (1.0 - cos_sim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(mse.item())

    ratio_val = (sum(history[-10:]) / 10) / (sum(history[:10]) / 10)
    print(f"  tiny-subset (1 cached crop, 300 steps) mse ratio (last10/first10) = {ratio_val:.4f}")
    if ratio_val > 0.5:
        raise SystemExit(
            f"STOP CONDITION: pipeline does not reproduce prior tiny-subset overfit behavior "
            f"(ratio {ratio_val:.4f} > 0.5) -- diagnose before running the real experiment."
        )
    print("  PASSED -- pipeline reproduces prior-experiment behavior; proceeding to the real experiment.")


def main() -> None:
    torch.manual_seed(SEED)
    try:
        device = select_device(requested="auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")

    if not MEL_CACHE_PATH.exists():
        raise SystemExit(f"{MEL_CACHE_PATH} not found -- run scripts/precompute_spectrogram_cache.py first")
    if not DAC_CACHE_PATH.exists():
        raise SystemExit(f"{DAC_CACHE_PATH} not found -- this experiment reuses it from experiment/muq-to-dac-more-data")

    mel_cache = torch.load(MEL_CACHE_PATH, map_location=device, weights_only=False)
    dac_cache = torch.load(DAC_CACHE_PATH, map_location=device, weights_only=False)
    mel_features = {label: v.to(device) for label, v in mel_cache["crops"].items()}
    dac_targets = {label: v["dac"].to(device) for label, v in dac_cache["crops"].items()}
    print(f"loaded {len(mel_features)} spectrogram crops (n_mels={mel_cache['n_mels']}, "
          f"hop_length={mel_cache['hop_length']}) and {len(dac_targets)} cached DAC targets (reused, unchanged)")
    assert set(mel_features) == set(dac_targets), "STOP CONDITION: spectrogram/DAC cache label mismatch"
    for label in mel_features:
        assert mel_features[label].shape[1] == dac_targets[label].shape[1], (
            f"STOP CONDITION: frame-count mismatch for {label}: "
            f"spectrogram={mel_features[label].shape[1]} dac={dac_targets[label].shape[1]}"
        )

    specs = load_manifest()
    by_split: dict[str, list[CropSpec]] = {"train": [], "val": [], "test": []}
    for spec in specs:
        by_split[spec.split].append(spec)
    for split, split_specs in by_split.items():
        print(f"  {split}: {len(split_specs)} crops, {len({s.track_id for s in split_specs})} tracks")

    track_splits: dict[str, set[str]] = {}
    for spec in specs:
        track_splits.setdefault(spec.track_id, set()).add(spec.split)
    for track_id, splits_seen in track_splits.items():
        assert len(splits_seen) == 1, f"STOP CONDITION: {track_id} spans splits {splits_seen}"

    run_tiny_subset_sanity_check(mel_features, dac_targets, device)

    dac = FrozenDacEncoder(DAC_MODEL_ID, device)  # decoder only, from here on

    print("\n--- Building SpectrogramToDacTranslator (fresh weights; TemporalConvNet unchanged from Gate 3) ---")
    torch.manual_seed(SEED)
    model = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    projection_param_count = sum(p.numel() for p in model.input_projection.parameters())
    print(f"parameters: {param_count:,} (of which {projection_param_count:,} in the input projection)")
    print(f"receptive field: {model.receptive_field_frames} frames at DAC's timeline")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()
    train_labels = [s.label for s in by_split["train"]]
    val_labels = [s.label for s in by_split["val"]]

    def compute_full_mse_and_cos(labels: list[str], m: SpectrogramToDacTranslator) -> tuple[float, float]:
        mse_total, cos_total = 0.0, 0.0
        with torch.no_grad():
            for label in labels:
                predicted = m(mel_features[label])
                mse_total += mse_loss_fn(predicted, dac_targets[label]).item()
                cos_total += nn.functional.cosine_similarity(predicted, dac_targets[label], dim=-1).mean().item()
        return mse_total / len(labels), cos_total / len(labels)

    print(f"\n--- Training ({STEPS} steps, MSE + {COSINE_LOSS_WEIGHT}*cosine, unchanged from experiment/muq-to-dac-more-data) ---")
    run_root = REPO_ROOT / "runs" / RUN_ID
    train_start = time.perf_counter()
    best_val_mse = float("inf")
    curves: list[dict[str, float]] = []
    window_mse: list[float] = []
    window_cos: list[float] = []
    pred_norm_window: list[float] = []
    target_norm_window: list[float] = []
    pred_var_window: list[float] = []
    target_var_window: list[float] = []
    for step in range(1, STEPS + 1):
        label = train_labels[step % len(train_labels)]
        predicted = model(mel_features[label])
        target = dac_targets[label]

        mse = mse_loss_fn(predicted, target)
        cos_sim = nn.functional.cosine_similarity(predicted, target, dim=-1).mean()
        total_loss = mse + COSINE_LOSS_WEIGHT * (1.0 - cos_sim)

        if not torch.isfinite(total_loss):
            raise SystemExit(f"STOP CONDITION: non-finite loss at step {step}: {total_loss.item()}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        window_mse.append(mse.item())
        window_cos.append(cos_sim.item())
        with torch.no_grad():
            pred_norm_window.append(predicted.norm(dim=-1).mean().item())
            target_norm_window.append(target.norm(dim=-1).mean().item())
            pred_var_window.append(predicted.var(dim=1).mean().item())
            target_var_window.append(target.var(dim=1).mean().item())

        if step % VAL_EVERY == 0 or step == STEPS:
            val_mse, val_cos = compute_full_mse_and_cos(val_labels, model)
            train_mse_window = sum(window_mse) / len(window_mse)
            train_cos_window = sum(window_cos) / len(window_cos)
            curves.append(
                {
                    "step": step,
                    "train_mse": train_mse_window,
                    "train_cos": train_cos_window,
                    "val_mse": val_mse,
                    "val_cos": val_cos,
                    "mean_pred_norm": sum(pred_norm_window) / len(pred_norm_window),
                    "mean_target_norm": sum(target_norm_window) / len(target_norm_window),
                    "mean_pred_variance": sum(pred_var_window) / len(pred_var_window),
                    "mean_target_variance": sum(target_var_window) / len(target_var_window),
                }
            )
            window_mse, window_cos = [], []
            pred_norm_window, target_norm_window, pred_var_window, target_var_window = [], [], [], []
            if val_mse < best_val_mse or step == VAL_EVERY:
                best_val_mse = val_mse
                _save_atomic(
                    run_root / "checkpoints" / "best.pt",
                    {"step": step, "model_state_dict": model.state_dict(), "val_mse": val_mse, "val_cos": val_cos},
                )
            if step % LOG_EVERY == 0 or step == STEPS:
                print(
                    f"  step {step}/{STEPS}: train_mse(window)={train_mse_window:.6f} "
                    f"train_cos(window)={train_cos_window:.4f} val_mse={val_mse:.6f} val_cos={val_cos:.4f}"
                )

    train_elapsed = time.perf_counter() - train_start
    peak_mem_mb = torch.mps.driver_allocated_memory() / 1e6 if device.type == "mps" else None
    print(f"\ntraining time: {train_elapsed:.2f}s ({train_elapsed / STEPS * 1000:.2f} ms/step)")
    print(f"best validation MSE during training: {best_val_mse:.6f}")
    if peak_mem_mb is not None:
        print(f"MPS driver-allocated memory at end of training: {peak_mem_mb:.1f} MB")

    print("\n--- Loading best-validation-MSE checkpoint for final comparison (not final step) ---")
    best_state = torch.load(run_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    best_model = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
    best_model.load_state_dict(best_state["model_state_dict"])
    best_model.eval()
    print(f"selected checkpoint: step {best_state['step']}, val_mse={best_state['val_mse']:.6f}")

    final_train_mse, final_train_cos = compute_full_mse_and_cos(train_labels, best_model)
    print(f"final full-train-set: mse={final_train_mse:.6f} cos={final_train_cos:.4f}")

    print("\n--- Evaluating all splits under all 3 conditions ---")
    all_labels = [s.label for s in specs]
    shuffle_pairing = {label: all_labels[(i + 1) % len(all_labels)] for i, label in enumerate(all_labels)}
    stft_loss_fn = MultiResolutionSTFTLoss(STFT_FFT_SIZES, STFT_HOP_SIZES, STFT_WIN_LENGTHS).to(device)

    artifact_root = REPO_ROOT / "artifacts" / RUN_ID
    metrics: dict[str, dict[str, float]] = {}
    latent_diagnostics: dict[str, dict[str, float]] = {}
    eval_start = time.perf_counter()
    for split in ("train", "val", "test"):
        split_specs = by_split[split]
        agg = {
            k: []
            for k in [
                "mse_correct",
                "cos_correct",
                "mse_shuffled",
                "cos_shuffled",
                "mse_shifted",
                "cos_shifted",
                "wav_corr_correct_vs_oracle",
                "wav_corr_shuffled_vs_oracle",
                "wav_corr_shifted_vs_oracle",
                "stft_correct_vs_oracle",
                "stft_shuffled_vs_oracle",
                "stft_shifted_vs_oracle",
                "wav_corr_oracle_vs_ref",
            ]
        }
        predicted_frames: list[np.ndarray] = []
        oracle_frames: list[np.ndarray] = []
        listening_tracks_seen: set[str] = set()
        n_listening = LISTENING_VAL_TRACKS if split == "val" else (LISTENING_TEST_TRACKS if split == "test" else 0)

        for spec in split_specs:
            label = spec.label
            target = dac_targets[label]
            mel_feat = mel_features[label]
            shuffled_mel = mel_features[shuffle_pairing[label]]

            with torch.no_grad():
                pred_correct = best_model(mel_feat)
                pred_shuffled = best_model(shuffled_mel)
                shifted_mel = shifted_time_control(mel_feat, TIME_SHIFT_FRAMES)
                pred_shifted = best_model(shifted_mel)

            agg["mse_correct"].append(mse_loss_fn(pred_correct, target).item())
            agg["cos_correct"].append(nn.functional.cosine_similarity(pred_correct, target, dim=-1).mean().item())
            agg["mse_shuffled"].append(mse_loss_fn(pred_shuffled, target).item())
            agg["cos_shuffled"].append(nn.functional.cosine_similarity(pred_shuffled, target, dim=-1).mean().item())
            agg["mse_shifted"].append(mse_loss_fn(pred_shifted, target).item())
            agg["cos_shifted"].append(nn.functional.cosine_similarity(pred_shifted, target, dim=-1).mean().item())

            predicted_frames.append(pred_correct.squeeze(0).detach().cpu().numpy())
            oracle_frames.append(target.squeeze(0).detach().cpu().numpy())

            reference = load_crop_waveform(spec, SAMPLE_RATE).squeeze(0)
            oracle_audio = dac.decode(target.transpose(1, 2)).squeeze(0).squeeze(0)
            correct_audio = dac.decode(pred_correct.transpose(1, 2)).squeeze(0).squeeze(0)
            shuffled_audio = dac.decode(pred_shuffled.transpose(1, 2)).squeeze(0).squeeze(0)
            shifted_audio = dac.decode(pred_shifted.transpose(1, 2)).squeeze(0).squeeze(0)

            n = min(len(reference), len(oracle_audio), len(correct_audio), len(shuffled_audio), len(shifted_audio))
            ref_n, oracle_n = reference[:n], oracle_audio[:n].cpu()
            correct_n = correct_audio[:n].cpu()
            shuffled_n, shifted_n = shuffled_audio[:n].cpu(), shifted_audio[:n].cpu()

            agg["wav_corr_correct_vs_oracle"].append(waveform_correlation(correct_n, oracle_n))
            agg["wav_corr_shuffled_vs_oracle"].append(waveform_correlation(shuffled_n, oracle_n))
            agg["wav_corr_shifted_vs_oracle"].append(waveform_correlation(shifted_n, oracle_n))
            agg["wav_corr_oracle_vs_ref"].append(waveform_correlation(oracle_n, ref_n))

            with torch.no_grad():
                oracle_b, correct_b = oracle_n.unsqueeze(0).to(device), correct_n.unsqueeze(0).to(device)
                shuffled_b, shifted_b = shuffled_n.unsqueeze(0).to(device), shifted_n.unsqueeze(0).to(device)
                agg["stft_correct_vs_oracle"].append(stft_loss_fn(correct_b, oracle_b)[0].item())
                agg["stft_shuffled_vs_oracle"].append(stft_loss_fn(shuffled_b, oracle_b)[0].item())
                agg["stft_shifted_vs_oracle"].append(stft_loss_fn(shifted_b, oracle_b)[0].item())

            if split in ("val", "test") and spec.crop_index == 0 and len(listening_tracks_seen) < n_listening:
                listening_tracks_seen.add(spec.track_id)
                step_tag = best_state["step"]
                for name, wav in [
                    ("reference", ref_n),
                    ("dac_oracle", oracle_n),
                    ("spectrogram_to_dac", correct_n),
                    ("shuffled_across", shuffled_n),
                    ("time_shifted", shifted_n),
                ]:
                    fname = f"{spec.label}_{name}_{RUN_ID}_step{step_tag}.wav"
                    _write_wav(artifact_root / fname, wav, SAMPLE_RATE)
                print(f"  wrote listening artifacts for {spec.label} ({split})")

        metrics[split] = {k: float(np.nanmean(v)) if v else float("nan") for k, v in agg.items()}

        pred_stats = compute_latent_stats(np.concatenate(predicted_frames, axis=0))
        oracle_stats = compute_latent_stats(np.concatenate(oracle_frames, axis=0))
        latent_diagnostics[split] = {
            "predicted": dataclasses.asdict(pred_stats),
            "oracle": dataclasses.asdict(oracle_stats),
            **ratio(pred_stats, oracle_stats),
        }
        print(f"  {split}: evaluated {len(split_specs)} crops")

    eval_elapsed = time.perf_counter() - eval_start
    print(f"evaluation phase took {eval_elapsed:.1f}s")

    comparison = None
    if MORE_DATA_REPORT_PATH.exists():
        with open(MORE_DATA_REPORT_PATH) as f:
            more_data_report = json.load(f)
        comparison = {
            "muq_to_dac_best_checkpoint_step": more_data_report["best_checkpoint_step"],
            "muq_to_dac_best_checkpoint_val_mse": more_data_report["best_checkpoint_val_mse"],
            "muq_to_dac_metrics_by_split": more_data_report["metrics_by_split"],
            "spectrogram_to_dac_best_checkpoint_step": best_state["step"],
            "spectrogram_to_dac_best_checkpoint_val_mse": best_state["val_mse"],
            "spectrogram_to_dac_metrics_by_split": metrics,
        }

    report = {
        "representation": "log_mel_spectrogram",
        "n_mels": N_MELS,
        "parameter_count": param_count,
        "input_projection_parameter_count": projection_param_count,
        "receptive_field_frames": model.receptive_field_frames,
        "training_time_seconds": train_elapsed,
        "ms_per_step": train_elapsed / STEPS * 1000,
        "evaluation_time_seconds": eval_elapsed,
        "peak_mps_driver_allocated_mb": peak_mem_mb,
        "best_checkpoint_step": best_state["step"],
        "best_checkpoint_val_mse": best_state["val_mse"],
        "final_full_train_mse": final_train_mse,
        "final_full_train_cos": final_train_cos,
        "num_train_crops": len(train_labels),
        "num_val_crops": len(val_labels),
        "num_test_crops": len(by_split["test"]),
        "num_train_tracks": len({s.track_id for s in by_split["train"]}),
        "num_val_tracks": len({s.track_id for s in by_split["val"]}),
        "num_test_tracks": len({s.track_id for s in by_split["test"]}),
        "training_curves": curves,
        "metrics_by_split": metrics,
        "latent_diagnostics_by_split": latent_diagnostics,
        "comparison_to_muq_to_dac_more_data": comparison,
    }
    report_path = artifact_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote full report to {report_path}")


if __name__ == "__main__":
    main()
