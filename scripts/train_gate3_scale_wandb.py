"""TASK-05: Gate-3 training-set-size scaling curve.

Bridges TASK-04's N=2 (perfect memorization) and TASK-01/02's N~15000 (norm collapse) by training
the exact same `SpectrogramToDacTranslator` / MSE+0.1*cosine / Adam(lr=1e-3, betas=(0.9,0.999))
pipeline at five nested training-set sizes (2, 8, 32, 128, 512 -- `data/manifests/
gate3_scale_{N}.csv`, `scripts/build_gate3_scale_manifests.py`), one level per run, all reusing
TASK-02's dense cache directly (no new cache).

**This is a training-set-size scaling experiment, not a generalization or optimization
experiment** -- per this task's explicit "do not focus on validation performance / do not attempt
to solve the task / do not redesign the architecture" instructions. All five levels use IDENTICAL
hyperparameters; the only variable is N.

## Step budget: fixed across all levels, NOT scaled with N (a deliberate, load-bearing choice)

Unlike TASK-02 (which scaled steps with crop count specifically to preserve per-crop *exposure*
and avoid confounding "more data" with "less trained"), this experiment needs the OPPOSITE: a
FIXED total step budget across all five levels, so that per-example exposure genuinely decreases
as N grows -- exactly the variable this task's three hypotheses (gradual collapse / sharp
threshold / no collapse at all) are about. Scaling steps with N here would let every example
memorize deeply regardless of N, trivially producing "no collapse at any N" by construction,
which would answer nothing. `MAX_STEPS=20000` per level (documented Condition C below), chosen
because TASK-04's N=2 case reached near-perfect memorization by ~step 2500 of 3000 -- 20000 is a
comfortable multiple of that, large enough to let every level fully saturate if it's going to.

## Stopping criterion, documented per this task's explicit requirement

Training stops at the first of: **(A)** best mean training MSE improves by less than 1e-5 for
`PATIENCE_STEPS` (5000) consecutive steps (saturation), or **(C)** `MAX_STEPS` (20000) reached.
Condition B (this task's "no meaningful improvement for a long period") is subsumed by A's
patience window. Checked every `VAL_EVERY` (250) steps.

## Per-step cost kept independent of N (critical for N=512 to be tractable)

Per-step logging (loss components, gradient norms) touches only the ONE example trained that
step -- same cost at every N. The full-N latent-only diagnostic pass (mean/median/min/max across
all N examples: mse, cosine, norm ratio, variance ratio, lag-1 autocorrelation -- no DAC decode)
runs only every `VAL_EVERY` steps. Decode-based metrics (waveform correlation, STFT distance)
run every `VAL_EVERY` steps too, but ONLY on a fixed 2-example subset (TASK-04's own
`pop.00012`/`disco.00049`, present in every level by construction) -- comparable across all five
levels and never O(N).

Findings and full write-up: notes/DECISIONS.md#task05-gate3-scaling, notes/SCRATCHPAD.md.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

import wandb
from muq_inversion.dac_encoder import FrozenDacEncoder
from muq_inversion.gate3_spectrogram_common import (
    TRACKS,
    aggregate_metrics_across_examples,
    count_nonfinite,
    evaluate_prediction,
    load_examples_from_manifest,
    three_way_grad_norms,
)
from muq_inversion.logging_utils import (
    DeviceSelectionError,
    capture_environment_manifest,
    select_device,
)
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.models.spectrogram_translator import SpectrogramToDacTranslator
from muq_inversion.spectrogram_encoder import N_MELS
from muq_inversion.wandb_logging import init_run, log_artifact_file, make_run_name

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 24000
DAC_MODEL_ID = "descript/dac_24khz"

# Identical to TASK-04 -- see module docstring for why the step budget itself is NOT scaled.
LEVELS = [2, 8, 32, 128, 512]
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
COSINE_LOSS_WEIGHT = 0.1
SEED = 0
MAX_STEPS = 20000
VAL_EVERY = 250
LOG_EVERY = 1000  # console print cadence
PATIENCE_STEPS = 5000
PATIENCE_MIN_DELTA = 1e-5

STFT_FFT_SIZES = (512, 1024, 2048)
STFT_HOP_SIZES = (50, 120, 240)
STFT_WIN_LENGTHS = (240, 600, 1200)

FIXED_AUDIO_SUBSET = TRACKS  # TASK-04's pop.00012/disco.00049 -- present in every level


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


def train_one_level(level: int, device: torch.device, environment: dict, wandb_mode: str) -> dict:
    print(f"\n{'=' * 70}\nLevel N={level}\n{'=' * 70}")
    manifest_path = REPO_ROOT / f"data/manifests/gate3_scale_{level}.csv"
    examples = load_examples_from_manifest(manifest_path, device)
    track_ids = list(examples.keys())
    assert len(track_ids) == level, f"STOP CONDITION: loaded {len(track_ids)} examples, expected {level}"
    for fixed_track in FIXED_AUDIO_SUBSET:
        assert fixed_track in examples, f"STOP CONDITION: fixed audio-subset track {fixed_track} missing from level {level}"

    dac = FrozenDacEncoder(DAC_MODEL_ID, device)
    stft_loss_fn = MultiResolutionSTFTLoss(STFT_FFT_SIZES, STFT_HOP_SIZES, STFT_WIN_LENGTHS).to(device)

    config = {
        "task": "TASK-05-gate3-scaling",
        "level_n": level,
        "learning_rate": LEARNING_RATE,
        "adam_betas": ADAM_BETAS,
        "cosine_loss_weight": COSINE_LOSS_WEIGHT,
        "max_steps": MAX_STEPS,
        "patience_steps": PATIENCE_STEPS,
        "val_every": VAL_EVERY,
        "seed": SEED,
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "num_examples": level,
        "wandb_version": wandb.__version__,
    }
    run = init_run(
        config=config,
        environment=environment,
        run_name=make_run_name(prefix=f"gate3_scale_{level}"),
        mode=wandb_mode,
        group="gate3-scaling",
        job_type="scaling-curve",
        tags=["spectrogram", "gate3", "scaling", f"n={level}"],
    )

    torch.manual_seed(SEED)
    model = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()

    run_root = REPO_ROOT / "runs" / "task05_gate3_scale" / f"n{level}"
    artifact_root = REPO_ROOT / "artifacts" / "task05_gate3_scale" / f"n{level}"

    def full_latent_eval() -> dict[str, dict[str, float]]:
        per_example = {}
        with torch.no_grad():
            for track_id, ex in examples.items():
                pred = model(ex.mel)
                per_example[track_id] = evaluate_prediction(pred, ex.dac, dac=None, stft_loss_fn=None, device=device, compute_audio_metrics=False)
        return per_example

    def fixed_subset_audio_eval() -> dict[str, dict[str, float]]:
        per_example = {}
        with torch.no_grad():
            for track_id in FIXED_AUDIO_SUBSET:
                ex = examples[track_id]
                pred = model(ex.mel)
                per_example[track_id] = evaluate_prediction(pred, ex.dac, dac=dac, stft_loss_fn=stft_loss_fn, device=device, compute_audio_metrics=True)
        return per_example

    best_mse = float("inf")
    best_step_since_improvement = 0
    scaling_curve: list[dict] = []
    train_start = time.perf_counter()
    step = 0
    stop_reason = "max_steps"
    while step < MAX_STEPS:
        step += 1
        track_id = track_ids[step % level]
        ex = examples[track_id]

        predicted = model(ex.mel)
        mse = mse_loss_fn(predicted, ex.dac)
        cos_sim = nn.functional.cosine_similarity(predicted, ex.dac, dim=-1).mean()
        total_loss = mse + COSINE_LOSS_WEIGHT * (1.0 - cos_sim)

        if not torch.isfinite(total_loss):
            raise SystemExit(f"STOP CONDITION: non-finite loss at level {level} step {step}: {total_loss.item()}")

        optimizer.zero_grad()
        total_loss.backward()
        grad_norms = three_way_grad_norms(model)
        all_grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        global_grad_norm = float(sum(g.norm(2) ** 2 for g in all_grads) ** 0.5)
        nonfinite_grads = count_nonfinite(all_grads)
        optimizer.step()

        run.log(
            {
                "train/total_loss": total_loss.item(),
                "train/mse": mse.item(),
                "train/cosine": cos_sim.item(),
                "grad/global_l2_norm": global_grad_norm,
                "grad/mel_projection_l2_norm": grad_norms["mel_projection_l2_norm"],
                "grad/translator_l2_norm": grad_norms["translator_l2_norm"],
                "grad/output_layer_l2_norm": grad_norms["output_layer_l2_norm"],
                "grad/nonfinite_count": nonfinite_grads,
            },
            step=step,
        )

        if step % VAL_EVERY == 0 or step == MAX_STEPS:
            per_example_latent = full_latent_eval()
            agg = aggregate_metrics_across_examples(per_example_latent)
            fixed_subset = fixed_subset_audio_eval()
            fixed_agg = aggregate_metrics_across_examples(fixed_subset)

            record = {
                "step": step,
                "mse_mean": agg["mse"]["mean"],
                "mse_median": agg["mse"]["median"],
                "mse_worst": agg["mse"]["max"],
                "cosine_mean": agg["cosine"]["mean"],
                "cosine_median": agg["cosine"]["median"],
                "cosine_worst": agg["cosine"]["min"],
                "norm_ratio_mean": agg["norm_ratio"]["mean"],
                # "worst" for a ratio-to-1.0 metric means farthest from 1.0; collapse in this
                # project has meant ratio -> 0 in every prior experiment, never > 1, so the
                # minimum is the informative "worst" direction here -- not a symmetric distance.
                "norm_ratio_worst": agg["norm_ratio"]["min"],
                "variance_ratio_mean": agg["variance_ratio"]["mean"],
                "variance_ratio_worst": agg["variance_ratio"]["min"],
                "lag1_autocorr_prediction_mean": agg["lag1_autocorrelation_prediction"]["mean"],
                "lag1_autocorr_oracle_mean": agg["lag1_autocorrelation_oracle"]["mean"],
                "waveform_correlation_to_oracle_mean": fixed_agg["waveform_correlation_to_oracle"]["mean"],
                "stft_distance_to_oracle_mean": fixed_agg["stft_distance_to_oracle"]["mean"],
            }
            scaling_curve.append(record)

            run.log(
                {
                    "collapse/norm_ratio_mean": agg["norm_ratio"]["mean"],
                    "collapse/variance_ratio_mean": agg["variance_ratio"]["mean"],
                    "collapse/lag1_autocorrelation_mean": agg["lag1_autocorrelation_prediction"]["mean"],
                    "temporal/frame_difference_mse_mean": agg["frame_difference_mse"]["mean"],
                    "temporal/oracle_autocorrelation_mean": agg["lag1_autocorrelation_oracle"]["mean"],
                    "audio/waveform_correlation_to_oracle": fixed_agg["waveform_correlation_to_oracle"]["mean"],
                    "audio/stft_distance_to_oracle": fixed_agg["stft_distance_to_oracle"]["mean"],
                    "eval/mse_mean": agg["mse"]["mean"],
                    "eval/mse_median": agg["mse"]["median"],
                    "eval/mse_worst": agg["mse"]["max"],
                    "eval/cosine_mean": agg["cosine"]["mean"],
                    "eval/cosine_median": agg["cosine"]["median"],
                    "eval/cosine_worst": agg["cosine"]["min"],
                },
                step=step,
            )

            if agg["mse"]["mean"] < best_mse - PATIENCE_MIN_DELTA:
                best_mse = agg["mse"]["mean"]
                best_step_since_improvement = step
                _save_atomic(
                    run_root / "checkpoints" / "best_by_mse.pt",
                    {"step": step, "model_state_dict": model.state_dict(), "mse_mean": best_mse},
                )

            if step % LOG_EVERY == 0 or step == MAX_STEPS:
                print(f"  step {step}/{MAX_STEPS}: mse_mean={agg['mse']['mean']:.6f} "
                      f"cosine_mean={agg['cosine']['mean']:.4f} norm_ratio_mean={agg['norm_ratio']['mean']:.4f}")

            if step - best_step_since_improvement >= PATIENCE_STEPS:
                stop_reason = "patience_plateau"
                print(f"  stopping: no MSE improvement > {PATIENCE_MIN_DELTA} in {PATIENCE_STEPS} steps")
                break

    train_elapsed = time.perf_counter() - train_start
    print(f"  level N={level} finished: {step} steps ({stop_reason}), {train_elapsed:.2f}s")

    # Final checkpoint + verified reload.
    final_ckpt_path = run_root / "checkpoints" / "final.pt"
    _save_atomic(final_ckpt_path, {"step": step, "model_state_dict": model.state_dict()})
    reloaded = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
    reloaded.load_state_dict(torch.load(final_ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        a = model(examples[track_ids[0]].mel)
        b = reloaded(examples[track_ids[0]].mel)
    assert torch.allclose(a, b, atol=1e-6), f"STOP CONDITION: level {level} final checkpoint reload mismatch"

    # Final full evaluation at the best-by-MSE checkpoint (matches TASK-04's "reload, don't trust
    # the last in-memory state" discipline).
    best_state = torch.load(run_root / "checkpoints" / "best_by_mse.pt", map_location=device, weights_only=False)
    best_model = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
    best_model.load_state_dict(best_state["model_state_dict"])
    best_model.eval()

    final_per_example_latent = {}
    with torch.no_grad():
        for track_id, ex in examples.items():
            pred = best_model(ex.mel)
            final_per_example_latent[track_id] = evaluate_prediction(pred, ex.dac, dac=None, stft_loss_fn=None, device=device, compute_audio_metrics=False)
    final_agg = aggregate_metrics_across_examples(final_per_example_latent)

    final_fixed_subset = {}
    audio_examples_for_log = {}
    with torch.no_grad():
        for track_id in FIXED_AUDIO_SUBSET:
            ex = examples[track_id]
            pred = best_model(ex.mel)
            final_fixed_subset[track_id] = evaluate_prediction(pred, ex.dac, dac=dac, stft_loss_fn=stft_loss_fn, device=device, compute_audio_metrics=True)
            pred_audio = dac.decode(pred.transpose(1, 2)).squeeze(0).squeeze(0).cpu()
            oracle_audio = dac.decode(ex.dac.transpose(1, 2)).squeeze(0).squeeze(0).cpu()
            _write_wav(artifact_root / f"{track_id}_n{level}_reconstruction.wav", pred_audio, SAMPLE_RATE)
            _write_wav(artifact_root / f"{track_id}_n{level}_dac_oracle.wav", oracle_audio, SAMPLE_RATE)
            audio_examples_for_log[f"{track_id}_reconstruction"] = (pred_audio.numpy(), SAMPLE_RATE)
    run.log(
        {f"audio_final/{k}": wandb.Audio(v[0], sample_rate=v[1], caption=f"N={level} final") for k, v in audio_examples_for_log.items()},
        step=step,
    )
    final_fixed_agg = aggregate_metrics_across_examples(final_fixed_subset)

    level_report = {
        "level_n": level,
        "parameter_count": param_count,
        "steps_run": step,
        "stop_reason": stop_reason,
        "training_time_seconds": train_elapsed,
        "best_checkpoint_step": best_state["step"],
        "scaling_curve": scaling_curve,
        "final_aggregate_latent_metrics": final_agg,
        "final_aggregate_fixed_subset_audio_metrics": final_fixed_agg,
        "final_per_example_latent_metrics": final_per_example_latent,
    }
    report_path = artifact_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(level_report, f, indent=2)
    log_artifact_file(run, report_path, f"report-n{level}", "report")
    log_artifact_file(run, run_root / "checkpoints" / "best_by_mse.pt", f"checkpoint-n{level}-best", "model")

    run.finish()
    return level_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default=",".join(str(x) for x in LEVELS), help="comma-separated list of N values to run")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    try:
        device = select_device(requested="auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")
    environment = capture_environment_manifest(REPO_ROOT)

    all_reports: dict[int, dict] = {}
    for level in levels:
        all_reports[level] = train_one_level(level, device, environment, args.wandb_mode)

    print("\n=== Scaling summary (final, best-by-MSE checkpoint per level) ===")
    header = f"{'N':>5s} {'MSE':>10s} {'cosine':>8s} {'norm_ratio':>11s} {'var_ratio':>10s} {'lag1_ac':>8s} {'wav_corr':>9s} {'stft':>8s}"
    print(header)
    summary_rows = []
    for level in levels:
        r = all_reports[level]
        lat = r["final_aggregate_latent_metrics"]
        aud = r["final_aggregate_fixed_subset_audio_metrics"]
        row = {
            "n": level,
            "mse_mean": lat["mse"]["mean"],
            "cosine_mean": lat["cosine"]["mean"],
            "norm_ratio_mean": lat["norm_ratio"]["mean"],
            "variance_ratio_mean": lat["variance_ratio"]["mean"],
            "lag1_autocorr_prediction_mean": lat["lag1_autocorrelation_prediction"]["mean"],
            "waveform_correlation_to_oracle_mean": aud["waveform_correlation_to_oracle"]["mean"],
            "stft_distance_to_oracle_mean": aud["stft_distance_to_oracle"]["mean"],
        }
        summary_rows.append(row)
        print(f"{level:>5d} {row['mse_mean']:>10.6f} {row['cosine_mean']:>8.4f} {row['norm_ratio_mean']:>11.4f} "
              f"{row['variance_ratio_mean']:>10.4f} {row['lag1_autocorr_prediction_mean']:>8.4f} "
              f"{row['waveform_correlation_to_oracle_mean']:>9.4f} {row['stft_distance_to_oracle_mean']:>8.4f}")

    summary_path = REPO_ROOT / "artifacts/task05_gate3_scale/scaling_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"levels": levels, "rows": summary_rows, "full_reports": all_reports}, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
