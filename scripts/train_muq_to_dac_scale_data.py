"""Data-scaling experiment: does the Gate 4 generalization failure resolve with ~10-20x more
track-disjoint training data, using the exact Gate 3 architecture/hyperparameters unchanged?

    frozen MuQ final hidden state -> existing TemporalConvNet -> DAC continuous latent
        -> frozen DAC decoder -> reconstructed audio

Nothing about the model, optimizer, loss, or frozen encoders/decoder changes from Gate 3
(`experiment/muq-to-dac-temporal-conv`, tagged `muq-to-dac-temporal-conv-v1`) or Gate 4
(`experiment/muq-to-dac-minimal-gate4`). Only the data scale changes: 80 train / 10 val / 10 test
track-disjoint GTZAN tracks (see `scripts/build_scale_data_manifest.py` ->
`data/manifests/scale_data_crops.csv`), with MuQ features and DAC latents precomputed once via
`scripts/precompute_scale_data_cache.py` and cached to
`data/cache/scale_data/muq_dac_cache.pt` -- this script never calls either encoder.

Step budget increased from Gate 4's 8000 to 20000 (documented deviation, same rationale as
Gate 4's own increase from Gate 3's 3000): 320 training crops need more total exposures than
Gate 4's 16 to reach a comparable per-example training depth. Per-step cost is unchanged (same
model, same batch size of 1), so this remains a fast, laptop-friendly run.

Findings and full write-up: notes/DECISIONS.md#muq-to-dac-scale-data, notes/SCRATCHPAD.md.
"""

from __future__ import annotations

import csv
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
from muq_inversion.logging_utils import DeviceSelectionError, select_device
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.models.temporal_conv import TemporalConvNet

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/scale_data_crops.csv"
CACHE_PATH = REPO_ROOT / "data/cache/scale_data/muq_dac_cache.pt"

SAMPLE_RATE = 24000
DAC_MODEL_ID = "descript/dac_24khz"  # only used for decoding at evaluation time, not encoding

# Unchanged from Gate 3 / Gate 4.
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
COSINE_LOSS_WEIGHT = 0.1
STEPS = 20000  # documented deviation from Gate 4's 8000 -- see module docstring
LOG_EVERY = 2000
VAL_EVERY = 500
SEED = 0
TIME_SHIFT_FRAMES = 10  # unchanged from Gate 4

STFT_FFT_SIZES = (512, 1024, 2048)
STFT_HOP_SIZES = (50, 120, 240)
STFT_WIN_LENGTHS = (240, 600, 1200)

RUN_ID = "scale_data"
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
                    experiment="scale_data",
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


def run_tiny_subset_sanity_check(muq_latents: dict, dac_targets: dict, frame_ratio: int, device: torch.device) -> None:
    """Mirrors Gate 3/Gate 4's pre-flight check: a fresh model overfitting ONE cached crop must
    reproduce Gate 3's fast convergence, verifying this script's training loop and the cached
    tensors are wired correctly before trusting the real, much slower, full-dataset run."""
    print("\n--- Pre-flight: tiny-subset sanity check (must reproduce Gate 3 overfit pace) ---")
    label = next(iter(muq_latents))
    muq_feat, dac_target = muq_latents[label], dac_targets[label]

    torch.manual_seed(SEED)
    model = TemporalConvNet(channels=muq_feat.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()
    history: list[float] = []
    for _ in range(300):
        repeated = muq_feat.repeat_interleave(frame_ratio, dim=1)
        predicted = model(repeated)
        mse = mse_loss_fn(predicted, dac_target)
        cos_sim = nn.functional.cosine_similarity(predicted, dac_target, dim=-1).mean()
        loss = mse + COSINE_LOSS_WEIGHT * (1.0 - cos_sim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(mse.item())

    ratio = (sum(history[-10:]) / 10) / (sum(history[:10]) / 10)
    print(f"  tiny-subset (1 cached crop, 300 steps) mse ratio (last10/first10) = {ratio:.4f}")
    if ratio > 0.5:
        raise SystemExit(
            f"STOP CONDITION: pipeline does not reproduce Gate 3's tiny-subset overfit behavior "
            f"(ratio {ratio:.4f} > 0.5) -- diagnose before running the real experiment."
        )
    print("  PASSED -- pipeline reproduces Gate 3 behavior; proceeding to the real experiment.")


def main() -> None:
    torch.manual_seed(SEED)
    try:
        device = select_device(requested="auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")

    if not CACHE_PATH.exists():
        raise SystemExit(f"{CACHE_PATH} not found -- run scripts/precompute_scale_data_cache.py first")
    cached = torch.load(CACHE_PATH, map_location=device, weights_only=False)
    frame_ratio = cached["frame_ratio"]
    muq_latents = {label: v["muq"].to(device) for label, v in cached["crops"].items()}
    dac_targets = {label: v["dac"].to(device) for label, v in cached["crops"].items()}
    print(f"loaded cache: {len(muq_latents)} crops, frame_ratio={frame_ratio}")

    specs = load_manifest()
    by_split: dict[str, list[CropSpec]] = {"train": [], "val": [], "test": []}
    for spec in specs:
        by_split[spec.split].append(spec)
    for split, split_specs in by_split.items():
        print(f"  {split}: {len(split_specs)} crops, {len({s.track_id for s in split_specs})} tracks")

    # Stop condition: verify track-disjointness at run time too, not just at manifest-build time.
    track_splits: dict[str, set[str]] = {}
    for spec in specs:
        track_splits.setdefault(spec.track_id, set()).add(spec.split)
    for track_id, splits_seen in track_splits.items():
        assert len(splits_seen) == 1, f"STOP CONDITION: {track_id} spans splits {splits_seen}"

    run_tiny_subset_sanity_check(muq_latents, dac_targets, frame_ratio, device)

    dac = FrozenDacEncoder(DAC_MODEL_ID, device)  # decoder only, from here on

    print("\n--- Building TemporalConvNet (fresh weights, architecture unchanged from Gate 3) ---")
    torch.manual_seed(SEED)
    channels = next(iter(muq_latents.values())).shape[-1]
    model = TemporalConvNet(channels=channels).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"parameters: {param_count:,}")
    print(f"receptive field: {model.receptive_field_frames} frames at DAC's timeline")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()
    train_labels = [s.label for s in by_split["train"]]
    val_labels = [s.label for s in by_split["val"]]

    def compute_mse_and_cos(labels: list[str], m: TemporalConvNet) -> tuple[float, float]:
        mse_total, cos_total = 0.0, 0.0
        with torch.no_grad():
            for label in labels:
                repeated = muq_latents[label].repeat_interleave(frame_ratio, dim=1)
                predicted = m(repeated)
                mse_total += mse_loss_fn(predicted, dac_targets[label]).item()
                cos_total += nn.functional.cosine_similarity(predicted, dac_targets[label], dim=-1).mean().item()
        return mse_total / len(labels), cos_total / len(labels)

    print(f"\n--- Training ({STEPS} steps, MSE + {COSINE_LOSS_WEIGHT}*cosine, unchanged from Gate 3) ---")
    run_root = REPO_ROOT / "runs" / RUN_ID
    train_start = time.perf_counter()
    best_val_mse = float("inf")
    curves: list[dict[str, float]] = []
    for step in range(1, STEPS + 1):
        label = train_labels[step % len(train_labels)]
        repeated = muq_latents[label].repeat_interleave(frame_ratio, dim=1)
        predicted = model(repeated)

        mse = mse_loss_fn(predicted, dac_targets[label])
        cos_sim = nn.functional.cosine_similarity(predicted, dac_targets[label], dim=-1).mean()
        total_loss = mse + COSINE_LOSS_WEIGHT * (1.0 - cos_sim)

        if not torch.isfinite(total_loss):
            raise SystemExit(f"STOP CONDITION: non-finite loss at step {step}: {total_loss.item()}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % VAL_EVERY == 0 or step == STEPS:
            val_mse, val_cos = compute_mse_and_cos(val_labels, model)
            train_mse, train_cos = compute_mse_and_cos(train_labels, model)
            curves.append({"step": step, "train_mse": train_mse, "train_cos": train_cos, "val_mse": val_mse, "val_cos": val_cos})
            if val_mse < best_val_mse or step == VAL_EVERY:
                best_val_mse = val_mse
                _save_atomic(
                    run_root / "checkpoints" / "best.pt",
                    {"step": step, "model_state_dict": model.state_dict(), "val_mse": val_mse, "val_cos": val_cos},
                )
            if step % LOG_EVERY == 0 or step == STEPS:
                print(
                    f"  step {step}/{STEPS}: train_mse={train_mse:.6f} train_cos={train_cos:.4f} "
                    f"val_mse={val_mse:.6f} val_cos={val_cos:.4f}"
                )

    train_elapsed = time.perf_counter() - train_start
    peak_mem_mb = torch.mps.driver_allocated_memory() / 1e6 if device.type == "mps" else None
    print(f"\ntraining time: {train_elapsed:.2f}s ({train_elapsed / STEPS * 1000:.2f} ms/step)")
    print(f"best validation MSE during training: {best_val_mse:.6f}")
    if peak_mem_mb is not None:
        print(f"MPS driver-allocated memory at end of training: {peak_mem_mb:.1f} MB")

    print("\n--- Loading best-validation-MSE checkpoint for final comparison (not final step) ---")
    best_state = torch.load(run_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    best_model = TemporalConvNet(channels=channels).to(device)
    best_model.load_state_dict(best_state["model_state_dict"])
    best_model.eval()
    print(f"selected checkpoint: step {best_state['step']}, val_mse={best_state['val_mse']:.6f}")

    print("\n--- Evaluating all splits under all 3 conditions ---")
    all_labels = [s.label for s in specs]
    shuffle_pairing = {label: all_labels[(i + 1) % len(all_labels)] for i, label in enumerate(all_labels)}
    stft_loss_fn = MultiResolutionSTFTLoss(STFT_FFT_SIZES, STFT_HOP_SIZES, STFT_WIN_LENGTHS).to(device)

    artifact_root = REPO_ROOT / "artifacts" / RUN_ID
    metrics: dict[str, dict[str, float]] = {}
    for split in ("train", "val", "test"):
        split_specs = by_split[split]
        agg = {k: [] for k in [
            "mse_correct", "cos_correct", "mse_shuffled", "cos_shuffled", "mse_shifted", "cos_shifted",
            "wav_corr_correct_vs_oracle", "wav_corr_shuffled_vs_oracle", "wav_corr_shifted_vs_oracle",
            "stft_correct_vs_oracle", "stft_shuffled_vs_oracle", "stft_shifted_vs_oracle",
            "wav_corr_oracle_vs_ref",
        ]}
        # Listening set: first N distinct tracks in val/test, one crop (crop_index 0) each.
        listening_tracks_seen: set[str] = set()
        n_listening = LISTENING_VAL_TRACKS if split == "val" else (LISTENING_TEST_TRACKS if split == "test" else 0)

        for spec in split_specs:
            label = spec.label
            target = dac_targets[label]
            muq_feat = muq_latents[label]
            shuffled_muq = muq_latents[shuffle_pairing[label]]

            with torch.no_grad():
                pred_correct = best_model(muq_feat.repeat_interleave(frame_ratio, dim=1))
                pred_shuffled = best_model(shuffled_muq.repeat_interleave(frame_ratio, dim=1))
                shifted_muq = shifted_time_control(muq_feat, TIME_SHIFT_FRAMES)
                pred_shifted = best_model(shifted_muq.repeat_interleave(frame_ratio, dim=1))

            agg["mse_correct"].append(mse_loss_fn(pred_correct, target).item())
            agg["cos_correct"].append(nn.functional.cosine_similarity(pred_correct, target, dim=-1).mean().item())
            agg["mse_shuffled"].append(mse_loss_fn(pred_shuffled, target).item())
            agg["cos_shuffled"].append(nn.functional.cosine_similarity(pred_shuffled, target, dim=-1).mean().item())
            agg["mse_shifted"].append(mse_loss_fn(pred_shifted, target).item())
            agg["cos_shifted"].append(nn.functional.cosine_similarity(pred_shifted, target, dim=-1).mean().item())

            reference = load_crop_waveform(spec, SAMPLE_RATE).squeeze(0)
            oracle_audio = dac.decode(target.transpose(1, 2)).squeeze(0).squeeze(0)
            correct_audio = dac.decode(pred_correct.transpose(1, 2)).squeeze(0).squeeze(0)
            shuffled_audio = dac.decode(pred_shuffled.transpose(1, 2)).squeeze(0).squeeze(0)
            shifted_audio = dac.decode(pred_shifted.transpose(1, 2)).squeeze(0).squeeze(0)

            n = min(len(reference), len(oracle_audio), len(correct_audio), len(shuffled_audio), len(shifted_audio))
            ref_n, oracle_n = reference[:n], oracle_audio[:n].cpu()
            correct_n, shuffled_n, shifted_n = correct_audio[:n].cpu(), shuffled_audio[:n].cpu(), shifted_audio[:n].cpu()

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
                    ("reference", ref_n), ("dac_oracle", oracle_n), ("temporal_conv", correct_n),
                    ("shuffled_across", shuffled_n), ("time_shifted", shifted_n),
                ]:
                    fname = f"{spec.label}_{name}_{RUN_ID}_step{step_tag}.wav"
                    _write_wav(artifact_root / fname, wav, SAMPLE_RATE)
                print(f"  wrote listening artifacts for {spec.label} ({split})")

        metrics[split] = {k: float(np.nanmean(v)) if v else float("nan") for k, v in agg.items()}

    report = {
        "parameter_count": param_count,
        "receptive_field_frames": model.receptive_field_frames,
        "training_time_seconds": train_elapsed,
        "ms_per_step": train_elapsed / STEPS * 1000,
        "peak_mps_driver_allocated_mb": peak_mem_mb,
        "best_checkpoint_step": best_state["step"],
        "best_checkpoint_val_mse": best_state["val_mse"],
        "num_train_crops": len(train_labels),
        "num_val_crops": len(val_labels),
        "num_test_crops": len(by_split["test"]),
        "training_curves": curves,
        "metrics_by_split": metrics,
    }
    report_path = artifact_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote full report to {report_path}")


if __name__ == "__main__":
    main()
