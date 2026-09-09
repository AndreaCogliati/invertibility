"""Gate 4: minimal generalization experiment for the MuQ-to-DAC temporal-conv pipeline.

    frozen MuQ final hidden state -> existing TemporalConvNet -> DAC continuous latent
        -> frozen DAC decoder -> reconstructed audio

Gate 3 (`experiment/muq-to-dac-temporal-conv`, tagged `muq-to-dac-temporal-conv-v1`) established
near-perfect memorization of two fixed 2-second crops (cosine similarity 0.9999, confirmed by
listening). This experiment asks whether that result reflects a transferable MuQ-to-DAC mapping
or memorization of those two specific crops, via two independent generalization tests:

    Experiment A: unseen crops from the SAME two tracks (crop-level generalization)
    Experiment B: unseen, entirely different tracks (track-level generalization)

No architecture, hidden size, receptive field, loss function, optimizer, or encoder/decoder
checkpoint is changed from the Gate 3 validated configuration -- only the data (more crops, more
tracks, explicit train/val/test splits) and the evaluation protocol (controls, held-out metrics)
are new. See `data/manifests/gate4_crops.csv` (built by `scripts/build_gate4_manifests.py`) for
the exact, checksummed split definition.

**Deviation from a literal reading of the prompt, documented explicitly:** the model is trained
FRESH per experiment on that experiment's own training crops, rather than reusing Gate 3's
checkpoint (which was optimized to memorize exactly `pop.00012`'s and `disco.00049`'s *original*
2-second crops -- reusing those weights would make the "unseen crop" evaluation in Experiment A
partially non-blind, and wouldn't apply at all to Experiment B's entirely different tracks). This
is the standard and only coherent way to run a train/val/test generalization test: train on the
train split, evaluate on held-out splits. The architecture, hyperparameters, and training
procedure are otherwise identical to Gate 3's validated configuration.

Findings and full write-up: notes/DECISIONS.md#muq-to-dac-minimal-gate4, notes/SCRATCHPAD.md.
"""

from __future__ import annotations

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
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.gate4_common import (
    CropSpec,
    load_crop_waveform,
    load_gate4_manifest,
    shifted_time_control,
    waveform_correlation,
)
from muq_inversion.logging_utils import DeviceSelectionError, select_device
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.models.temporal_conv import TemporalConvNet

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/gate4_crops.csv"

SAMPLE_RATE = 24000
MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
DAC_MODEL_ID = "descript/dac_24khz"

# Unchanged from Gate 3 (experiment/muq-to-dac-temporal-conv, tagged muq-to-dac-temporal-conv-v1).
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
COSINE_LOSS_WEIGHT = 0.1
# Increased from Gate 3's 3000 (documented deviation): training now covers 16 diverse crops per
# experiment instead of 1-2 memorized ones, a harder joint-fitting problem needing more exposures
# per example to reach a comparable convergence pace. Per-step cost is unchanged and fast
# (~7ms/step measured), so this remains trivial next to any HiFi-GAN-family experiment.
STEPS = 8000
LOG_EVERY = 500
VAL_EVERY = 200
SEED = 0
TIME_SHIFT_FRAMES = 10  # fixed nonzero offset for control #3, ~400ms at MuQ's 25 Hz

# Multi-resolution STFT distance config reused verbatim from configs/smoke.yaml
# (notes/DECISIONS.md#gate3-loss-configs), used here only as an evaluation metric -- not part of
# the training objective, which stays MSE + cosine per this experiment's constraints.
STFT_FFT_SIZES = (512, 1024, 2048)
STFT_HOP_SIZES = (50, 120, 240)
STFT_WIN_LENGTHS = (240, 600, 1200)

RUN_ID = "gate4_minimal"


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


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise SystemExit(f"STOP CONDITION: non-finite values in {name}")


def run_tiny_subset_sanity_check(muq: FrozenMuQEncoder, dac: FrozenDacEncoder, device: torch.device) -> None:
    """Stop-condition pre-flight check: "training loss does not reproduce the Gate 3 overfit
    behavior on a tiny subset." Deliberately mirrors Gate 3 exactly -- a fresh model overfitting
    ONE crop -- rather than checking convergence speed on the full multi-crop training set
    (which is a harder, more diverse problem expected to converge more slowly; slower
    convergence there is data about generalization difficulty, not evidence of a broken
    pipeline). This check exists to catch a genuine regression in the encoders/model code before
    trusting the real experiments' results."""
    print("\n--- Pre-flight: tiny-subset sanity check (must reproduce Gate 3 overfit pace) ---")
    specs = load_gate4_manifest(MANIFEST_PATH, "A")
    spec = next(s for s in specs if s.split == "train" and s.crop_index == 0 and s.track_id == "pop.00012")
    waveform = load_crop_waveform(spec, SAMPLE_RATE)
    muq_feat = muq.encode(waveform).detach().clone()
    dac_target = dac.encode(waveform).transpose(1, 2).detach().clone()
    frame_ratio = round(dac_target.shape[1] / muq_feat.shape[1])

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
    print(f"  tiny-subset (1 crop, 300 steps) mse ratio (last10/first10) = {ratio:.4f}")
    if ratio > 0.5:
        raise SystemExit(
            f"STOP CONDITION: pipeline does not reproduce Gate 3's tiny-subset overfit behavior "
            f"(ratio {ratio:.4f} > 0.5) -- this indicates a regression in encoders/model code, "
            f"not a data-scale generalization issue. Diagnose before running the real experiments."
        )
    print("  PASSED -- pipeline reproduces Gate 3 behavior; proceeding to the real experiments.")


def run_experiment(
    experiment: str,
    muq: FrozenMuQEncoder,
    dac: FrozenDacEncoder,
    device: torch.device,
    artifact_root: Path,
    run_root: Path,
) -> dict:
    print(f"\n{'=' * 70}\nExperiment {experiment}\n{'=' * 70}")
    specs = load_gate4_manifest(MANIFEST_PATH, experiment)
    by_split: dict[str, list[CropSpec]] = {"train": [], "val": [], "test": []}
    for spec in specs:
        by_split[spec.split].append(spec)
    for split, split_specs in by_split.items():
        tracks = sorted({s.track_id for s in split_specs})
        print(f"  {split}: {len(split_specs)} crops, tracks={tracks}")

    # Stop condition: no track appears in more than one split (Experiment B) -- verified again
    # here at run time, not just at manifest-build time, in case the manifest was hand-edited.
    track_splits: dict[str, set[str]] = {}
    for spec in specs:
        track_splits.setdefault(spec.track_id, set()).add(spec.split)
    for track_id, splits_seen in track_splits.items():
        if experiment == "B":
            assert len(splits_seen) == 1, f"STOP CONDITION: {track_id} spans splits {splits_seen}"

    print("\n--- Precomputing MuQ (input) and DAC (target) latents for every crop ---")
    muq_latents: dict[str, torch.Tensor] = {}
    dac_targets: dict[str, torch.Tensor] = {}
    frame_ratio: int | None = None
    for spec in specs:
        waveform = load_crop_waveform(spec, SAMPLE_RATE)
        muq_feat = muq.encode(waveform)  # (1, num_frames, 1024)
        dac_feat = dac.encode(waveform)  # (1, 1024, num_frames)
        ratio = dac_feat.shape[-1] / muq_feat.shape[1]
        assert ratio == round(ratio), (
            f"STOP CONDITION: non-integer frame ratio for {spec.label}: {ratio}"
        )
        ratio = round(ratio)
        if frame_ratio is None:
            frame_ratio = ratio
        assert ratio == frame_ratio, (
            f"STOP CONDITION: frame ratio changed across crops: {ratio} != {frame_ratio}"
        )
        _assert_finite(f"muq_latent[{spec.label}]", muq_feat)
        _assert_finite(f"dac_target[{spec.label}]", dac_feat)
        # .clone(): encode() runs under torch.inference_mode(); cached tensors need a real clone
        # to be usable in later autograd (see experiment/muq-to-dac-mlp's SCRATCHPAD entry).
        muq_latents[spec.label] = muq_feat.detach().clone()
        dac_targets[spec.label] = dac_feat.transpose(1, 2).detach().clone()  # -> (1, T, 1024)
    assert frame_ratio is not None
    print(f"frame_ratio={frame_ratio} (verified identical for all {len(specs)} crops)")

    print("\n--- Building TemporalConvNet (fresh weights, architecture unchanged from Gate 3) ---")
    torch.manual_seed(SEED)
    model = TemporalConvNet(channels=muq_latents[specs[0].label].shape[-1]).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"parameters: {param_count:,}")
    print(f"receptive field: {model.receptive_field_frames} frames at DAC's timeline")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()
    train_labels = [s.label for s in by_split["train"]]
    val_labels = [s.label for s in by_split["val"]]

    def compute_mse(labels: list[str], m: TemporalConvNet) -> float:
        if not labels:
            return float("nan")
        total = 0.0
        with torch.no_grad():
            for label in labels:
                repeated = muq_latents[label].repeat_interleave(frame_ratio, dim=1)
                predicted = m(repeated)
                total += mse_loss_fn(predicted, dac_targets[label]).item()
        return total / len(labels)

    print(f"\n--- Training ({STEPS} steps, MSE + {COSINE_LOSS_WEIGHT}*cosine, unchanged from Gate 3) ---")
    train_start = time.perf_counter()
    best_val_mse = float("inf")
    train_mse_history: list[float] = []
    nonfinite_seen = False
    for step in range(1, STEPS + 1):
        label = train_labels[step % len(train_labels)]
        muq_feat = muq_latents[label]
        target = dac_targets[label]
        repeated = muq_feat.repeat_interleave(frame_ratio, dim=1)
        predicted = model(repeated)

        mse = mse_loss_fn(predicted, target)
        cos_sim = nn.functional.cosine_similarity(predicted, target, dim=-1).mean()
        total_loss = mse + COSINE_LOSS_WEIGHT * (1.0 - cos_sim)

        if not torch.isfinite(total_loss):
            nonfinite_seen = True
            raise SystemExit(f"STOP CONDITION: non-finite loss at step {step}: {total_loss.item()}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_mse_history.append(mse.item())

        if step == 300:
            # Informational only, not a stop condition: fitting 16 diverse training crops jointly
            # is a harder problem than Gate 3's 1-2-crop overfit (already verified separately by
            # `run_tiny_subset_sanity_check` before this experiment started), so slower
            # convergence here is expected data about generalization difficulty, not a failure.
            ratio = (sum(train_mse_history[-10:]) / 10) / (sum(train_mse_history[:10]) / 10)
            print(f"  [informational @ step 300] train mse ratio (last10/first10) = {ratio:.4f}")

        if step % VAL_EVERY == 0 or step == STEPS:
            val_mse = compute_mse(val_labels, model)
            if val_mse < best_val_mse or step == VAL_EVERY:
                best_val_mse = val_mse
                _save_atomic(
                    run_root / experiment / "checkpoints" / "best.pt",
                    {"step": step, "model_state_dict": model.state_dict(), "val_mse": val_mse},
                )

        if step == 1 or step % LOG_EVERY == 0 or step == STEPS:
            print(
                f"  step {step}/{STEPS}: mse={mse.item():.6f} cos_sim={cos_sim.item():.4f} "
                f"val_mse={compute_mse(val_labels, model):.6f} clip={label}"
            )

    train_elapsed = time.perf_counter() - train_start
    peak_mem_mb = None
    if device.type == "mps":
        peak_mem_mb = torch.mps.driver_allocated_memory() / 1e6

    print(f"\ntraining time: {train_elapsed:.2f}s ({train_elapsed / STEPS * 1000:.2f} ms/step)")
    print(f"best validation MSE during training: {best_val_mse:.6f}")
    if peak_mem_mb is not None:
        print(f"MPS driver-allocated memory at end of training: {peak_mem_mb:.1f} MB")

    print("\n--- Loading best-validation-MSE checkpoint for final comparison (not final step) ---")
    best_state = torch.load(run_root / experiment / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    best_model = TemporalConvNet(channels=muq_latents[specs[0].label].shape[-1]).to(device)
    best_model.load_state_dict(best_state["model_state_dict"])
    best_model.eval()
    print(f"selected checkpoint: step {best_state['step']}, val_mse={best_state['val_mse']:.6f}")

    print("\n--- Evaluating all splits under all 4 conditions ---")
    all_labels = [s.label for s in specs]
    shuffle_pairing = {label: all_labels[(i + 1) % len(all_labels)] for i, label in enumerate(all_labels)}

    metrics: dict[str, dict[str, float]] = {}
    listening_done: set[str] = set()
    for split in ("train", "val", "test"):
        split_specs = by_split[split]
        if not split_specs:
            continue
        agg = {k: [] for k in [
            "mse_correct", "cos_correct", "mse_shuffled", "cos_shuffled",
            "mse_shifted", "cos_shifted", "wav_corr_correct_vs_oracle",
            "wav_corr_shuffled_vs_oracle", "wav_corr_shifted_vs_oracle",
            "stft_correct_vs_oracle", "stft_shuffled_vs_oracle", "stft_shifted_vs_oracle",
            "wav_corr_oracle_vs_ref", "stft_oracle_vs_ref",
        ]}
        stft_loss_fn = MultiResolutionSTFTLoss(STFT_FFT_SIZES, STFT_HOP_SIZES, STFT_WIN_LENGTHS).to(device)

        for spec in split_specs:
            label = spec.label
            target = dac_targets[label]
            muq_feat = muq_latents[label]
            shuffled_muq = muq_latents[shuffle_pairing[label]]

            with torch.no_grad():
                repeated_correct = muq_feat.repeat_interleave(frame_ratio, dim=1)
                pred_correct = best_model(repeated_correct)

                repeated_shuffled = shuffled_muq.repeat_interleave(frame_ratio, dim=1)
                pred_shuffled = best_model(repeated_shuffled)

                shifted_muq = shifted_time_control(muq_feat, TIME_SHIFT_FRAMES)
                repeated_shifted = shifted_muq.repeat_interleave(frame_ratio, dim=1)
                pred_shifted = best_model(repeated_shifted)

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
                ref_b = ref_n.unsqueeze(0).to(device)
                agg["stft_correct_vs_oracle"].append(stft_loss_fn(correct_b, oracle_b)[0].item())
                agg["stft_shuffled_vs_oracle"].append(stft_loss_fn(shuffled_b, oracle_b)[0].item())
                agg["stft_shifted_vs_oracle"].append(stft_loss_fn(shifted_b, oracle_b)[0].item())
                agg["stft_oracle_vs_ref"].append(stft_loss_fn(oracle_b, ref_b)[0].item())

            # Listening artifacts: one crop per split (the first), per this experiment's
            # "small fixed set" requirement.
            if split not in listening_done:
                listening_done.add(split)
                step_tag = best_state["step"]
                base = artifact_root / experiment
                for name, wav in [
                    ("reference", ref_n),
                    ("dac_oracle", oracle_n),
                    ("temporal_conv", correct_n),
                    ("shuffled_across", shuffled_n),
                    ("time_shifted", shifted_n),
                ]:
                    fname = f"{spec.label}_{name}_{RUN_ID}_step{step_tag}.wav"
                    _write_wav(base / fname, wav, SAMPLE_RATE)
                print(f"  wrote listening artifacts for {spec.label} ({split}) to {base}")

        metrics[split] = {k: float(np.nanmean(v)) if v else float("nan") for k, v in agg.items()}

    return {
        "experiment": experiment,
        "parameter_count": param_count,
        "receptive_field_frames": model.receptive_field_frames,
        "training_time_seconds": train_elapsed,
        "ms_per_step": train_elapsed / STEPS * 1000,
        "peak_mps_driver_allocated_mb": peak_mem_mb,
        "best_checkpoint_step": best_state["step"],
        "best_checkpoint_val_mse": best_state["val_mse"],
        "nonfinite_seen": nonfinite_seen,
        "metrics_by_split": metrics,
    }


def main() -> None:
    torch.manual_seed(SEED)
    try:
        device = select_device(requested="auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")

    muq = FrozenMuQEncoder(MUQ_MODEL_ID, device)
    dac = FrozenDacEncoder(DAC_MODEL_ID, device)

    run_tiny_subset_sanity_check(muq, dac, device)

    artifact_root = REPO_ROOT / "artifacts" / RUN_ID
    run_root = REPO_ROOT / "runs" / RUN_ID

    report = {"a": run_experiment("A", muq, dac, device, artifact_root, run_root),
              "b": run_experiment("B", muq, dac, device, artifact_root, run_root)}

    report_path = artifact_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote full report to {report_path}")


if __name__ == "__main__":
    main()
