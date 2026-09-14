"""TASK-03: exact reproduction of TASK-02 (dense-coverage spectrogram-to-DAC) with comprehensive
Weights & Biases instrumentation, runnable on either MPS (this machine) or CUDA (BlueHive).

    log-mel spectrogram (dense, ~100% GTZAN coverage) -> input projection -> TemporalConvNet
        -> DAC continuous latent -> frozen DAC decoder -> reconstructed audio

**Nothing about the scientific formulation changes from `train_spectrogram_to_dac_dense.py`
(TASK-02)** -- same manifest/cache, same architecture, same loss, same optimizer/LR/betas, same
step budget (450000, unless overridden via `--steps` for a short infrastructure-validation run --
see below), same checkpoint-selection rule, same evaluation protocol. This script is additive
instrumentation, not a redesign, per this task's explicit "do not modify the scientific
formulation" / "any deviation must be documented and justified" instructions.

## Deliberate deviations, documented

1. **STFT distance and waveform correlation to the DAC oracle are computed every validation cycle
   only on a FIXED small subset (the same 6 tracks used for audio logging), not the full
   val/test set.** TASK-02's own full-split decode-based evaluation took 3805s for ONE pass over
   all three splits; doing that at every one of 450 validation checkpoints would take on the
   order of days. Full-split STFT/waveform-correlation numbers (matching TASK-02's own protocol
   exactly) are still computed once, at the end, for the final report and reproduction table.
2. **Audio logging (waveforms to W&B) happens every `AUDIO_LOG_EVERY` steps (10x
   `VAL_EVERY`, i.e. every 10000 steps), not every validation cycle**, to control W&B storage
   and decode cost -- still frequent enough to "listen to training evolution" as this task asks.
3. **Gradient statistics are logged every `GRAD_LOG_EVERY` steps (every 100 training steps)**,
   not literally every one of 450000 steps, which would make W&B's own ingestion the wall-time
   bottleneck. "Every training interval" is interpreted as this fixed, documented cadence.
4. **`--steps` CLI override exists ONLY for infrastructure validation** (proving the full W&B
   pipeline -- config panel, gradient tracking, checkpoint tables, audio logging, artifacts --
   works end-to-end before committing to the multi-hour BlueHive run). Any run intended as the
   actual TASK-02 reproduction must be launched with the default (no override).

Findings and full write-up: notes/DECISIONS.md#task03-bluehive-wandb, notes/SCRATCHPAD.md.
"""

from __future__ import annotations

import argparse
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
from muq_inversion.grad_diagnostics import (
    compute_gradient_stats,
    gradient_stats_to_wandb_dict,
)
from muq_inversion.latent_diagnostics import compute_latent_stats, ratio
from muq_inversion.logging_utils import (
    DeviceSelectionError,
    capture_environment_manifest,
    select_device,
)
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.models.spectrogram_translator import SpectrogramToDacTranslator
from muq_inversion.spectrogram_encoder import N_MELS
from muq_inversion.wandb_logging import (
    init_run,
    log_artifact_file,
    log_audio_examples,
    log_checkpoint_tables,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/dense_crops.csv"  # reused unchanged from TASK-02
CACHE_PATH = REPO_ROOT / "data/cache/dense/mel_dac_cache.pt"  # reused unchanged from TASK-02
TASK02_REPORT_PATH = REPO_ROOT / "artifacts/dense_spectrogram_control/report.json"

SAMPLE_RATE = 24000
DAC_MODEL_ID = "descript/dac_24khz"

# Unchanged from TASK-02.
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
COSINE_LOSS_WEIGHT = 0.1
SEED = 0
TIME_SHIFT_FRAMES = 10
DEFAULT_STEPS = 450000
LOG_EVERY = 37500
VAL_EVERY = 1000

# New for TASK-03's W&B instrumentation -- see module docstring's "deliberate deviations".
GRAD_LOG_EVERY = 100
AUDIO_LOG_EVERY = 10 * VAL_EVERY

STFT_FFT_SIZES = (512, 1024, 2048)
STFT_HOP_SIZES = (50, 120, 240)
STFT_WIN_LENGTHS = (240, 600, 1200)

RUN_ID = "task03_wandb_repro"
LISTENING_TRACKS = {
    "val": ["blues.00014", "blues.00029", "blues.00034"],
    "test": ["blues.00005", "blues.00024", "blues.00025"],
}
LISTENING_CROP_INDEX = 2  # t=4.0s, same as TASK-02


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
                    experiment="task03",
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


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e6
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return None


def run_tiny_subset_sanity_check(mel_features: dict, dac_targets: dict, device: torch.device) -> None:
    print("\n--- Pre-flight: tiny-subset sanity check ---")
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
    print(f"  tiny-subset mse ratio (last10/first10) = {ratio_val:.4f}")
    if ratio_val > 0.5:
        raise SystemExit(f"STOP CONDITION: sanity check failed (ratio {ratio_val:.4f} > 0.5)")
    print("  PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS,
        help=f"Training steps. DEFAULT ({DEFAULT_STEPS}) reproduces TASK-02 exactly -- override "
             f"ONLY for a short infrastructure-validation run, never for the real reproduction.",
    )
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--job-id", default=None, help="BlueHive/SLURM job ID, if not already in $SLURM_JOB_ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    is_smoke_test = args.steps != DEFAULT_STEPS
    if is_smoke_test:
        print(f"*** --steps={args.steps} != {DEFAULT_STEPS}: this is an infrastructure-validation "
              f"run, NOT a TASK-02 reproduction. Do not use its numbers for the reproduction table. ***")

    torch.manual_seed(SEED)
    try:
        device = select_device(requested="auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")

    environment = capture_environment_manifest(REPO_ROOT)
    print(f"environment: hostname={environment['hostname']} cuda={environment['cuda_available']} "
          f"mps={environment['mps_available']} slurm_job_id={environment['slurm_job_id']}")

    if not CACHE_PATH.exists():
        raise SystemExit(f"{CACHE_PATH} not found -- run scripts/precompute_dense_cache.py first")
    cache = torch.load(CACHE_PATH, map_location=device, weights_only=False)
    mel_features = {label: v["mel"].to(device) for label, v in cache["crops"].items()}
    dac_targets = {label: v["dac"].to(device) for label, v in cache["crops"].items()}
    print(f"loaded {len(mel_features)} crops")

    specs = load_manifest()
    by_split: dict[str, list[CropSpec]] = {"train": [], "val": [], "test": []}
    for spec in specs:
        by_split[spec.split].append(spec)
    dataset_stats = {
        f"num_{split}_crops": len(split_specs) for split, split_specs in by_split.items()
    } | {
        f"num_{split}_tracks": len({s.track_id for s in split_specs}) for split, split_specs in by_split.items()
    }
    for split, split_specs in by_split.items():
        print(f"  {split}: {len(split_specs)} crops, {len({s.track_id for s in split_specs})} tracks")

    run_tiny_subset_sanity_check(mel_features, dac_targets, device)

    dac = FrozenDacEncoder(DAC_MODEL_ID, device)

    torch.manual_seed(SEED)
    model = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()
    train_labels = [s.label for s in by_split["train"]]
    val_labels = [s.label for s in by_split["val"]]

    config = {
        "learning_rate": LEARNING_RATE,
        "adam_betas": ADAM_BETAS,
        "weight_decay": 0.0,
        "scheduler": None,
        "cosine_loss_weight": COSINE_LOSS_WEIGHT,
        "steps": args.steps,
        "val_every": VAL_EVERY,
        "batch_size": 1,
        "seed": SEED,
        "model_parameter_count": param_count,
        "n_mels": N_MELS,
        "dac_model_id": DAC_MODEL_ID,
        "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "is_smoke_test": is_smoke_test,
        **dataset_stats,
    }
    run = init_run(config=config, environment=environment, job_id=args.job_id, run_name=args.run_name, mode=args.wandb_mode)
    print(f"W&B run: {run.name if hasattr(run, 'name') else '(disabled)'}")

    def compute_full_mse_and_cos(labels: list[str], m: SpectrogramToDacTranslator) -> tuple[float, float]:
        mse_total, cos_total = 0.0, 0.0
        with torch.no_grad():
            for label in labels:
                predicted = m(mel_features[label])
                mse_total += mse_loss_fn(predicted, dac_targets[label]).item()
                cos_total += nn.functional.cosine_similarity(predicted, dac_targets[label], dim=-1).mean().item()
        return mse_total / len(labels), cos_total / len(labels)

    fixed_val_specs = [
        s for s in by_split["val"] if s.crop_index == LISTENING_CROP_INDEX and s.track_id in LISTENING_TRACKS["val"]
    ]
    fixed_test_specs = [
        s for s in by_split["test"] if s.crop_index == LISTENING_CROP_INDEX and s.track_id in LISTENING_TRACKS["test"]
    ]
    fixed_specs = fixed_val_specs + fixed_test_specs
    stft_loss_fn = MultiResolutionSTFTLoss(STFT_FFT_SIZES, STFT_HOP_SIZES, STFT_WIN_LENGTHS).to(device)

    def evaluate_fixed_subset(m: SpectrogramToDacTranslator) -> dict[str, float]:
        """STFT distance / waveform correlation on the small fixed subset only -- see module
        docstring's deviation #1."""
        stft_vals, wav_corr_vals = [], []
        with torch.no_grad():
            for spec in fixed_specs:
                label = spec.label
                target = dac_targets[label]
                pred = m(mel_features[label])
                oracle_audio = dac.decode(target.transpose(1, 2)).squeeze(0).squeeze(0)
                pred_audio = dac.decode(pred.transpose(1, 2)).squeeze(0).squeeze(0)
                n = min(len(oracle_audio), len(pred_audio))
                oracle_n, pred_n = oracle_audio[:n].cpu(), pred_audio[:n].cpu()
                wav_corr_vals.append(waveform_correlation(pred_n, oracle_n))
                stft_vals.append(
                    stft_loss_fn(pred_n.unsqueeze(0).to(device), oracle_n.unsqueeze(0).to(device))[0].item()
                )
        return {
            "stft_distance_fixed_subset": float(np.nanmean(stft_vals)),
            "wav_corr_fixed_subset": float(np.nanmean(wav_corr_vals)),
        }

    run_root = REPO_ROOT / "runs" / RUN_ID
    train_start = time.perf_counter()
    best_val_mse = float("inf")
    checkpoint_history: list[dict[str, float]] = []
    window_mse: list[float] = []
    window_cos: list[float] = []
    for step in range(1, args.steps + 1):
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

        if step % GRAD_LOG_EVERY == 0:
            grad_stats = compute_gradient_stats(model)
            run.log(gradient_stats_to_wandb_dict(grad_stats), step=step)

        optimizer.step()
        window_mse.append(mse.item())
        window_cos.append(cos_sim.item())

        run.log(
            {"loss/total": total_loss.item(), "loss/mse": mse.item(), "loss/cosine": (1.0 - cos_sim.item())},
            step=step,
        )

        if step % VAL_EVERY == 0 or step == args.steps:
            val_mse, val_cos = compute_full_mse_and_cos(val_labels, model)
            train_mse_window = sum(window_mse) / len(window_mse)
            train_cos_window = sum(window_cos) / len(window_cos)
            window_mse, window_cos = [], []

            predicted_frames, target_frames = [], []
            with torch.no_grad():
                for label in val_labels:
                    predicted_frames.append(model(mel_features[label]).squeeze(0).cpu().numpy())
                    target_frames.append(dac_targets[label].squeeze(0).cpu().numpy())
            pred_stats = compute_latent_stats(np.concatenate(predicted_frames, axis=0))
            target_stats = compute_latent_stats(np.concatenate(target_frames, axis=0))
            ratios = ratio(pred_stats, target_stats)

            record = {
                "step": step,
                "val_mse": val_mse,
                "val_cos": val_cos,
                "norm_ratio": ratios["norm_ratio"],
                "variance_ratio": ratios["variance_ratio"],
            }
            checkpoint_history.append(record)

            run.log(
                {
                    "val/mse": val_mse,
                    "val/cosine": val_cos,
                    "train_window/mse": train_mse_window,
                    "train_window/cosine": train_cos_window,
                    "prediction_stats/target_norm_mean": target_stats.vector_norm_mean,
                    "prediction_stats/target_norm_median": target_stats.vector_norm_median,
                    "prediction_stats/target_variance": target_stats.channel_variance_mean,
                    "prediction_stats/pred_norm_mean": pred_stats.vector_norm_mean,
                    "prediction_stats/pred_norm_median": pred_stats.vector_norm_median,
                    "prediction_stats/pred_variance": pred_stats.channel_variance_mean,
                    "prediction_stats/norm_ratio": ratios["norm_ratio"],
                    "prediction_stats/variance_ratio": ratios["variance_ratio"],
                    "temporal/lag1_autocorr_predicted": pred_stats.lag1_autocorr,
                    "temporal/lag1_autocorr_target": target_stats.lag1_autocorr,
                    "temporal/frame_to_frame_variance_predicted": pred_stats.frame_to_frame_variance_mean,
                    "temporal/frame_to_frame_variance_target": target_stats.frame_to_frame_variance_mean,
                },
                step=step,
            )

            if step % (5 * VAL_EVERY) == 0 or step == args.steps:
                fixed_metrics = evaluate_fixed_subset(model)
                run.log({f"val/{k}": v for k, v in fixed_metrics.items()}, step=step)

            if val_mse < best_val_mse or step == VAL_EVERY:
                best_val_mse = val_mse
                _save_atomic(
                    run_root / "checkpoints" / "best.pt",
                    {"step": step, "model_state_dict": model.state_dict(), "val_mse": val_mse, "val_cos": val_cos},
                )

            if step % AUDIO_LOG_EVERY == 0 or step == args.steps:
                examples = {}
                with torch.no_grad():
                    for spec in fixed_specs[:3]:  # cap W&B upload volume; 3 of the 6 fixed tracks
                        pred = model(mel_features[spec.label])
                        pred_audio = dac.decode(pred.transpose(1, 2)).squeeze(0).squeeze(0).cpu().numpy()
                        examples[f"{spec.split}_{spec.track_id}_prediction"] = (pred_audio, SAMPLE_RATE)
                log_audio_examples(run, step, examples)

            if step % LOG_EVERY == 0 or step == args.steps:
                print(f"  step {step}/{args.steps}: val_mse={val_mse:.6f} val_cos={val_cos:.4f} "
                      f"norm_ratio={ratios['norm_ratio']:.4f}")

    train_elapsed = time.perf_counter() - train_start
    print(f"\ntraining time: {train_elapsed:.2f}s")

    log_checkpoint_tables(run, checkpoint_history)
    if len(checkpoint_history) >= 3:
        steps_list = [r["step"] for r in checkpoint_history]
        mse_list = [r["val_mse"] for r in checkpoint_history]
        cos_list = [r["val_cos"] for r in checkpoint_history]
        norm_ratio_list = [r["norm_ratio"] for r in checkpoint_history]
        var_ratio_list = [r["variance_ratio"] for r in checkpoint_history]
        try:
            import wandb

            run.log(
                {
                    "scatter/val_mse_vs_norm_ratio": wandb.plot.scatter(
                        wandb.Table(data=list(zip(mse_list, norm_ratio_list, strict=True)), columns=["val_mse", "norm_ratio"]),
                        "val_mse", "norm_ratio",
                    ),
                    "scatter/val_mse_vs_variance_ratio": wandb.plot.scatter(
                        wandb.Table(data=list(zip(mse_list, var_ratio_list, strict=True)), columns=["val_mse", "variance_ratio"]),
                        "val_mse", "variance_ratio",
                    ),
                    "scatter/cosine_vs_step": wandb.plot.scatter(
                        wandb.Table(data=list(zip(steps_list, cos_list, strict=True)), columns=["step", "val_cos"]),
                        "step", "val_cos",
                    ),
                }
            )
        except ImportError:
            pass

    print("\n--- Loading best-validation-MSE checkpoint for final comparison ---")
    best_state = torch.load(run_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    best_model = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
    best_model.load_state_dict(best_state["model_state_dict"])
    best_model.eval()
    print(f"selected checkpoint: step {best_state['step']}, val_mse={best_state['val_mse']:.6f}")

    if is_smoke_test:
        eval_label = "--- Reduced-subset evaluation (smoke test: full-split eval SKIPPED for speed) ---"
        eval_subset_cap = 5  # a handful of crops per split -- enough to exercise every code path
    else:
        eval_label = "--- Full-split evaluation (matches TASK-02's exact protocol) ---"
        eval_subset_cap = None
    print(f"\n{eval_label}")
    all_labels = [s.label for s in specs]
    shuffle_pairing = {label: all_labels[(i + 1) % len(all_labels)] for i, label in enumerate(all_labels)}
    artifact_root = REPO_ROOT / "artifacts" / RUN_ID
    metrics: dict[str, dict[str, float]] = {}
    latent_diagnostics: dict[str, dict[str, float]] = {}
    eval_start = time.perf_counter()
    for split in ("train", "val", "test"):
        split_specs = by_split[split][:eval_subset_cap] if eval_subset_cap else by_split[split]
        agg = {k: [] for k in [
            "mse_correct", "cos_correct", "mse_shuffled", "cos_shuffled", "mse_shifted", "cos_shifted",
            "wav_corr_correct_vs_oracle", "wav_corr_shuffled_vs_oracle", "wav_corr_shifted_vs_oracle",
            "stft_correct_vs_oracle", "stft_shuffled_vs_oracle", "stft_shifted_vs_oracle", "wav_corr_oracle_vs_ref",
        ]}
        predicted_frames, oracle_frames = [], []
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

            if spec.crop_index == LISTENING_CROP_INDEX and spec.track_id in LISTENING_TRACKS.get(split, []):
                step_tag = best_state["step"]
                for name, wav in [("reference", ref_n), ("dac_oracle", oracle_n), ("spectrogram_to_dac", correct_n),
                                   ("shuffled_across", shuffled_n), ("time_shifted", shifted_n)]:
                    _write_wav(artifact_root / f"{spec.label}_{name}_{RUN_ID}_step{step_tag}.wav", wav, SAMPLE_RATE)

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
    peak_mem_mb = peak_memory_mb(device)

    comparison = {}
    if TASK02_REPORT_PATH.exists():
        with open(TASK02_REPORT_PATH) as f:
            task02_report = json.load(f)
        comparison["task02_m4_max"] = {
            "best_checkpoint_step": task02_report["best_checkpoint_step"],
            "best_checkpoint_val_mse": task02_report["best_checkpoint_val_mse"],
            "metrics_by_split": task02_report["metrics_by_split"],
        }
    comparison["this_run"] = {
        "best_checkpoint_step": best_state["step"],
        "best_checkpoint_val_mse": best_state["val_mse"],
        "metrics_by_split": metrics,
        "device": str(device),
        "hostname": environment["hostname"],
        "is_smoke_test": is_smoke_test,
    }

    report = {
        "run_id": RUN_ID,
        "device": str(device),
        "environment": environment,
        "config": config,
        "steps": args.steps,
        "is_smoke_test": is_smoke_test,
        "training_time_seconds": train_elapsed,
        "evaluation_time_seconds": eval_elapsed,
        "peak_memory_mb": peak_mem_mb,
        "best_checkpoint_step": best_state["step"],
        "best_checkpoint_val_mse": best_state["val_mse"],
        "num_train_crops": len(train_labels),
        "num_val_crops": len(val_labels),
        "num_test_crops": len(by_split["test"]),
        "metrics_by_split": metrics,
        "latent_diagnostics_by_split": latent_diagnostics,
        "checkpoint_history": checkpoint_history,
        "comparison": comparison,
    }
    report_path = artifact_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote full report to {report_path}")

    config_path = artifact_root / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    for path, name, artifact_type in [
        (run_root / "checkpoints" / "best.pt", "best-checkpoint", "model"),
        (report_path, "evaluation-report", "report"),
        (config_path, "training-config", "config"),
        (MANIFEST_PATH, "dense-crops-manifest", "dataset"),
    ]:
        if path.exists():
            log_artifact_file(run, path, name, artifact_type)

    run.finish()
    print("W&B run finished.")


if __name__ == "__main__":
    main()
