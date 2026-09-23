"""TASK-12: EnCodec N=32 compute ablation.

TASK-11 found that swapping the target codec from DAC to EnCodec substantially reduced (but did
not remove) the N=32 gap: EnCodec's translator recovered 79.2% of its own oracle ceiling at N=32
after a 20000-step budget, vs. DAC's 48.4%. This script holds N=32 fixed (the exact
`data/manifests/gate3_scale_32.csv` TASK-05/06/11 used, unchanged) and varies ONLY the training
budget -- 20k (TASK-11's own completed run, reused unchanged per the "preferred training
strategy: train independently to 20k/80k/320k" instruction), 80k, and 320k -- to test whether the
residual TASK-11 gap is compute-limited or reflects a deeper architecture/objective/information
limit. Mirrors TASK-06's DAC compute ablation (`train_gate3_n32_compute_ablation.py`) design
exactly, substituting EnCodec throughout.

## Early stopping disabled -- same deliberate deviation as TASK-06, for the same reason

TASK-05/11 used a 5000-step patience window tuned for a 20000-step budget; TASK-11's own N=32 run
triggered it exactly at the 20000-step boundary. Reusing that window for an 80000- or
320000-step budget would very likely stop the run near the same step count again, silently
preventing this experiment from ever testing whether more compute helps. This script runs the
FULL fixed budget unconditionally for the 80k/320k runs, no patience-based early exit -- the same
choice TASK-06 made for DAC, for the identical reason.

## 20k data point: TASK-11's own completed run, reused unchanged

Per this task's own "preferred training strategy" (train independently to each budget from the
same seed/init, not one extended trajectory with intermediate snapshots) and its "confirm TASK-11
outputs are preserved" reproducibility check, the 20k comparison point is TASK-11's actual N=32
report (`artifacts/task11_gate3_encodec_scale/n32/report.json`), not retrained here.

## Both TASK-11 fixes preserved, per this task's explicit instruction

- Freshly-extracted EnCodec targets are `.clone()`-d (`gate3_encodec_common.load_examples_from_manifest_encodec`,
  unmodified) to avoid the inference-tensor/autograd failure.
- DAC reference metrics in the final comparison are pulled directly from TASK-11's own stored,
  lag-aligned `dac_oracle_aggregate`/`dac_comparison` values -- never recomputed here, and never
  using a naive unaligned waveform comparison.

Findings and full write-up: notes/DECISIONS.md#task12-encodec-n32-compute-ablation, notes/SCRATCHPAD.md.
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
from muq_inversion.encodec_encoder import FrozenEncodecEncoder
from muq_inversion.gate3_encodec_common import load_examples_from_manifest_encodec
from muq_inversion.gate3_spectrogram_common import (
    aggregate_metrics_across_examples,
    count_nonfinite,
    evaluate_prediction,
    rank_examples_by_metric,
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
MANIFEST_PATH = REPO_ROOT / "data/manifests/gate3_scale_32.csv"  # reused unchanged from TASK-05/06/11
SAMPLE_RATE = 24000
ENCODEC_MODEL_ID = "facebook/encodec_24khz"
ENCODEC_BANDWIDTH = 24.0
ENCODEC_CHANNELS = 128

LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
COSINE_LOSS_WEIGHT = 0.1
SEED = 0

STFT_FFT_SIZES = (512, 1024, 2048)
STFT_HOP_SIZES = (50, 120, 240)
STFT_WIN_LENGTHS = (240, 600, 1200)

FIXED_AUDIO_SUBSET = ["pop.00012", "disco.00049"]  # present in every gate3_scale_* manifest
ALL_AUDIO_MILESTONES = [20000, 40000, 80000, 160000, 320000]

VAL_EVERY_BY_BUDGET = {80000: 1000, 320000: 4000}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, required=True, choices=[80000, 320000])
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budget = args.budget
    val_every = VAL_EVERY_BY_BUDGET[budget]

    try:
        device = select_device(requested="auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")
    environment = capture_environment_manifest(REPO_ROOT)

    encodec = FrozenEncodecEncoder(ENCODEC_MODEL_ID, device, bandwidth=ENCODEC_BANDWIDTH)
    examples = load_examples_from_manifest_encodec(MANIFEST_PATH, device, encodec, SAMPLE_RATE)
    track_ids = list(examples.keys())
    assert len(track_ids) == 32, f"STOP CONDITION: expected 32 examples, got {len(track_ids)}"
    for fixed_track in FIXED_AUDIO_SUBSET:
        assert fixed_track in examples, f"STOP CONDITION: {fixed_track} missing from N=32 manifest"

    stft_loss_fn = MultiResolutionSTFTLoss(STFT_FFT_SIZES, STFT_HOP_SIZES, STFT_WIN_LENGTHS).to(device)

    torch.manual_seed(SEED)
    model = SpectrogramToDacTranslator(input_channels=N_MELS, dac_channels=ENCODEC_CHANNELS).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"parameters: {param_count:,}  receptive_field: {model.receptive_field_frames} frames  "
          f"input: (1,150,{N_MELS})  output: (1,150,{ENCODEC_CHANNELS})")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()

    config = {
        "task": "TASK-12-encodec-n32-compute-ablation",
        "budget_steps": budget,
        "early_stopping": False,
        "learning_rate": LEARNING_RATE,
        "adam_betas": ADAM_BETAS,
        "cosine_loss_weight": COSINE_LOSS_WEIGHT,
        "seed": SEED,
        "val_every": val_every,
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "num_examples": 32,
        "parameter_count": param_count,
        "receptive_field_frames": model.receptive_field_frames,
        "encodec_model_id": ENCODEC_MODEL_ID,
        "encodec_bandwidth_kbps": ENCODEC_BANDWIDTH,
        "wandb_version": wandb.__version__,
    }
    run = init_run(
        config=config,
        environment=environment,
        run_name=make_run_name(prefix=f"encodec_n32_{budget // 1000}k"),
        mode=args.wandb_mode,
        group="gate3-encodec-n32-compute-ablation",
        job_type="compute-ablation",
        tags=["spectrogram", "gate3", "encodec", "n32", "compute-ablation", f"budget={budget}"],
    )

    run_root = REPO_ROOT / "runs" / "task12_encodec_n32_compute_ablation" / f"budget_{budget}"
    artifact_root = REPO_ROOT / "artifacts" / "task12_encodec_n32_compute_ablation" / f"budget_{budget}"
    audio_milestones = {m for m in ALL_AUDIO_MILESTONES if m <= budget}

    def full_eval(compute_audio: bool) -> dict[str, dict[str, float]]:
        per_example = {}
        with torch.no_grad():
            for track_id, ex in examples.items():
                pred = model(ex.mel)
                per_example[track_id] = evaluate_prediction(
                    pred, ex.target, dac=encodec if compute_audio else None, stft_loss_fn=stft_loss_fn if compute_audio else None,
                    device=device, compute_audio_metrics=compute_audio,
                )
        return per_example

    def export_best_median_worst(step: int, per_example: dict[str, dict[str, float]]) -> None:
        ranking = rank_examples_by_metric(per_example)  # ascending: worst first
        worst_id, median_id, best_id = ranking[0], ranking[len(ranking) // 2], ranking[-1]
        for label, track_id in [("worst", worst_id), ("median", median_id), ("best", best_id)]:
            ex = examples[track_id]
            with torch.no_grad():
                pred = model(ex.mel)
                pred_audio = encodec.decode(pred.transpose(1, 2)).squeeze(0).cpu()
            _write_wav(artifact_root / f"step{step}_{label}_{track_id}.wav", pred_audio, SAMPLE_RATE)
        print(f"    best/median/worst @ step {step}: {best_id} / {median_id} / {worst_id}")

    def export_fixed_subset_audio(step: int) -> None:
        for track_id in FIXED_AUDIO_SUBSET:
            ex = examples[track_id]
            with torch.no_grad():
                pred = model(ex.mel)
                pred_audio = encodec.decode(pred.transpose(1, 2)).squeeze(0).cpu()
                oracle_audio = encodec.decode(ex.target.transpose(1, 2)).squeeze(0).cpu()
            _write_wav(artifact_root / f"step{step}_{track_id}_reconstruction.wav", pred_audio, SAMPLE_RATE)
            if step == min(audio_milestones):
                _write_wav(artifact_root / f"{track_id}_encodec_oracle.wav", oracle_audio, SAMPLE_RATE)
                _write_wav(artifact_root / f"{track_id}_original.wav", ex.waveform.squeeze(0).cpu(), SAMPLE_RATE)
            run.log(
                {f"audio/{track_id}": wandb.Audio(pred_audio.numpy(), sample_rate=SAMPLE_RATE, caption=f"{track_id} | step {step}")},
                step=step,
            )

    checkpoint_history: list[dict] = []
    best_mse = float("inf")
    best_wav_corr = -float("inf")
    train_start = time.perf_counter()
    for step in range(1, budget + 1):
        track_id = track_ids[step % 32]
        ex = examples[track_id]

        predicted = model(ex.mel)
        mse = mse_loss_fn(predicted, ex.target)
        cos_sim = nn.functional.cosine_similarity(predicted, ex.target, dim=-1).mean()
        total_loss = mse + COSINE_LOSS_WEIGHT * (1.0 - cos_sim)

        if not torch.isfinite(total_loss):
            raise SystemExit(f"STOP CONDITION: non-finite loss at step {step}: {total_loss.item()}")

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

        if step % val_every == 0 or step == budget:
            is_audio_milestone = step in audio_milestones
            per_example = full_eval(compute_audio=True)
            agg = aggregate_metrics_across_examples(per_example)

            record = {"step": step, "per_example": per_example, "aggregate": agg}
            checkpoint_history.append(record)

            log_payload = {f"distribution/{metric}_{stat}": agg[metric][stat] for metric in agg for stat in agg[metric]}
            run.log(log_payload, step=step)

            if agg["mse"]["mean"] < best_mse:
                best_mse = agg["mse"]["mean"]
                _save_atomic(run_root / "checkpoints" / "best_by_mse.pt", {"step": step, "model_state_dict": model.state_dict(), "mse_mean": best_mse})

            wav_corr_now = agg["waveform_correlation_to_oracle"]["mean"]
            if wav_corr_now > best_wav_corr:
                best_wav_corr = wav_corr_now
                _save_atomic(run_root / "checkpoints" / "best_by_wav_corr.pt", {"step": step, "model_state_dict": model.state_dict(), "wav_corr_mean": best_wav_corr})

            print(f"  step {step}/{budget}: mse_mean={agg['mse']['mean']:.6f} cosine_mean={agg['cosine']['mean']:.4f} "
                  f"wav_corr_mean={wav_corr_now:.4f} "
                  f"wav_corr_min={agg['waveform_correlation_to_oracle']['min']:.4f} "
                  f"wav_corr_max={agg['waveform_correlation_to_oracle']['max']:.4f}")

            if is_audio_milestone:
                export_fixed_subset_audio(step)
                export_best_median_worst(step, per_example)

    train_elapsed = time.perf_counter() - train_start
    print(f"\nfinished: {budget} steps, {train_elapsed:.2f}s ({train_elapsed / budget * 1000:.2f} ms/step)")

    final_ckpt_path = run_root / "checkpoints" / "final.pt"
    _save_atomic(final_ckpt_path, {"step": budget, "model_state_dict": model.state_dict()})
    reloaded = SpectrogramToDacTranslator(input_channels=N_MELS, dac_channels=ENCODEC_CHANNELS).to(device)
    reloaded.load_state_dict(torch.load(final_ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        a = model(examples[track_ids[0]].mel)
        b = reloaded(examples[track_ids[0]].mel)
    assert torch.allclose(a, b, atol=1e-6), "STOP CONDITION: final checkpoint reload mismatch"

    def eval_checkpoint(state_dict) -> tuple[dict, dict]:
        m = SpectrogramToDacTranslator(input_channels=N_MELS, dac_channels=ENCODEC_CHANNELS).to(device)
        m.load_state_dict(state_dict)
        m.eval()
        per_example = {}
        with torch.no_grad():
            for track_id, ex in examples.items():
                pred = m(ex.mel)
                per_example[track_id] = evaluate_prediction(pred, ex.target, dac=encodec, stft_loss_fn=stft_loss_fn, device=device, compute_audio_metrics=True)
        return per_example, aggregate_metrics_across_examples(per_example)

    best_state = torch.load(run_root / "checkpoints" / "best_by_mse.pt", map_location=device, weights_only=False)
    best_per_example, best_agg = eval_checkpoint(best_state["model_state_dict"])
    final_per_example, final_agg = eval_checkpoint(model.state_dict())

    degraded = final_agg["waveform_correlation_to_oracle"]["mean"] < best_agg["waveform_correlation_to_oracle"]["mean"] - 0.02

    report = {
        "budget_steps": budget,
        "early_stopping": False,
        "val_every": val_every,
        "training_time_seconds": train_elapsed,
        "parameter_count": param_count,
        "receptive_field_frames": model.receptive_field_frames,
        "best_checkpoint_step": best_state["step"],
        "final_checkpoint_step": budget,
        "final_degraded_from_best": degraded,
        "checkpoint_history": checkpoint_history,
        "best_checkpoint": {"aggregate_metrics": best_agg, "per_example_metrics": best_per_example},
        "final_checkpoint": {"aggregate_metrics": final_agg, "per_example_metrics": final_per_example},
    }
    report_path = artifact_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {report_path}")

    log_artifact_file(run, report_path, f"report-encodec-budget{budget}", "report")
    log_artifact_file(run, run_root / "checkpoints" / "best_by_mse.pt", f"checkpoint-encodec-budget{budget}-best", "model")
    run.finish()
    print("W&B run finished.")


if __name__ == "__main__":
    main()
