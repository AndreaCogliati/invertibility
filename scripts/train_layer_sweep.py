"""MuQ hidden-layer sweep: does an earlier or intermediate MuQ hidden layer predict DAC latents
more effectively than the final hidden state (the baseline used throughout Gate 3 / Gate 4 / the
two data-scaling experiments)?

    frozen MuQ hidden layer L -> existing TemporalConvNet -> DAC continuous latent
        -> frozen DAC decoder -> reconstructed audio

Only the selected MuQ hidden layer changes across the 5 runs in this sweep. Everything else --
architecture, receptive field, hidden size, optimizer, learning rate, loss, DAC target
definition, MuQ checkpoint, GTZAN split, crop manifest, checkpoint-selection rule -- is identical
across all 5 layers and identical to `experiment/muq-to-dac-more-data`.

## Screening budget: 45000 steps (not the full 60000), chosen from the existing final-layer curve

`artifacts/more_data/report.json`'s training curve shows the turbulent, unstable early phase
ending around step 32000-33000 (val MSE dropped from spikes of 30-142 down into the 18-22 range),
then a comparatively stable region from ~33000 to 60000 (val MSE 17.3-21.4, val cosine 0.18-0.23,
best at step 50000). By step 45000 the curve has already sampled several points within ~1.3 MSE
of the eventual best (17.7 at step 47000 vs. 17.29 at step 50000) -- extending to 60000 mostly
re-confirms a plateau already visible by 45000, which is not useful information for a quick
screening sweep across 5 layers. 45000 steps was chosen BEFORE running any layer other than the
final-layer control, per this experiment's explicit requirement to document the budget choice
before examining other layers' results.

## Final-layer control-run sanity check

Before comparing any other layer, layer 12 (the final layer, i.e. this experiment's baseline) is
retrained at this reduced 45000-step budget and its best validation MSE/cosine are checked
against the "stable region" the original 60000-step run showed from step ~33000 onward (val MSE
in [17, 25], val cosine >= 0.15) -- generous bounds, since an exact numeric match isn't expected
under a different step budget and MPS's known non-determinism (notes/DECISIONS.md
#gate4-mps-resume-tolerance), but a run landing far outside this range would indicate something
is wrong with this script's pipeline, not with MuQ layer choice. If this check fails, the script
stops before training any other layer.

## Evaluation scope, to control wall time across 5 layers

Per-split full decode-based waveform correlation / multi-resolution STFT distance are computed
for validation and test only (not train) -- the metric spec asks for "predicted-to-DAC-oracle"
(singular "predicted", i.e. the correctly-conditioned prediction) values, not per-control decode
metrics; shuffled/time-shifted controls are evaluated via latent-space cosine margin only, which
needs no decoding. Train MSE/cosine are computed directly from cached latents (no decode either).
This keeps the 5-layer sweep's total evaluation cost proportional to 2x(188+226) decodes rather
than 2x(1584+188+226), which would dominate wall time five times over.

Findings and full write-up: notes/DECISIONS.md#muq-layer-sweep, notes/SCRATCHPAD.md.
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
    shifted_time_control,
    waveform_correlation,
)
from muq_inversion.layer_sweep_common import (
    LAYER_LABELS,
    CheckpointRecord,
    layer_cache_label,
    select_best_checkpoint,
    select_evenly_spaced_layers,
)
from muq_inversion.logging_utils import DeviceSelectionError, select_device
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.models.temporal_conv import TemporalConvNet

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/more_data_crops.csv"  # reused unchanged
LAYER_CACHE_PATH = REPO_ROOT / "data/cache/layer_sweep/muq_layers_cache.pt"
DAC_CACHE_PATH = REPO_ROOT / "data/cache/more_data/muq_dac_cache.pt"  # reused unchanged, read-only

SAMPLE_RATE = 24000
DAC_MODEL_ID = "descript/dac_24khz"
NUM_TRANSFORMER_LAYERS = 12
MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"

# Unchanged from every prior MuQ-to-DAC experiment.
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
COSINE_LOSS_WEIGHT = 0.1
STEPS = 45000  # documented screening budget -- see module docstring
LOG_EVERY = 5000
VAL_EVERY = 500
SEED = 0
TIME_SHIFT_FRAMES = 10

STFT_FFT_SIZES = (512, 1024, 2048)
STFT_HOP_SIZES = (50, 120, 240)
STFT_WIN_LENGTHS = (240, 600, 1200)

RUN_ID = "layer_sweep"
FINAL_LAYER_SANITY_VAL_MSE_MAX = 25.0
FINAL_LAYER_SANITY_VAL_COS_MIN = 0.15

CANDIDATE_INDICES = select_evenly_spaced_layers(NUM_TRANSFORMER_LAYERS, num_candidates=5)


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
                    experiment="layer_sweep",
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


def train_one_layer(
    layer_label: str,
    layer_idx: int,
    muq_latents: dict[str, torch.Tensor],
    dac_targets: dict[str, torch.Tensor],
    frame_ratio: int,
    by_split: dict[str, list[CropSpec]],
    device: torch.device,
    dac: FrozenDacEncoder,
) -> dict:
    print(f"\n{'=' * 70}\nLayer '{layer_label}' (index {layer_idx})\n{'=' * 70}")

    torch.manual_seed(SEED)
    channels = next(iter(muq_latents.values())).shape[-1]
    model = TemporalConvNet(channels=channels).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()

    train_labels = [s.label for s in by_split["train"]]
    val_labels = [s.label for s in by_split["val"]]

    def latent_mse_cos(labels: list[str], m: TemporalConvNet) -> tuple[float, float]:
        mse_total, cos_total = 0.0, 0.0
        with torch.no_grad():
            for label in labels:
                repeated = muq_latents[label].repeat_interleave(frame_ratio, dim=1)
                predicted = m(repeated)
                mse_total += mse_loss_fn(predicted, dac_targets[label]).item()
                cos_total += nn.functional.cosine_similarity(predicted, dac_targets[label], dim=-1).mean().item()
        return mse_total / len(labels), cos_total / len(labels)

    run_root = REPO_ROOT / "runs" / RUN_ID / layer_label
    train_start = time.perf_counter()
    checkpoint_history: list[CheckpointRecord] = []
    window_mse: list[float] = []
    window_cos: list[float] = []
    curves: list[dict[str, float]] = []
    for step in range(1, STEPS + 1):
        label = train_labels[step % len(train_labels)]
        repeated = muq_latents[label].repeat_interleave(frame_ratio, dim=1)
        predicted = model(repeated)

        mse = mse_loss_fn(predicted, dac_targets[label])
        cos_sim = nn.functional.cosine_similarity(predicted, dac_targets[label], dim=-1).mean()
        total_loss = mse + COSINE_LOSS_WEIGHT * (1.0 - cos_sim)

        if not torch.isfinite(total_loss):
            raise SystemExit(f"STOP CONDITION: non-finite loss at step {step} (layer {layer_label}): {total_loss.item()}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        window_mse.append(mse.item())
        window_cos.append(cos_sim.item())

        if step % VAL_EVERY == 0 or step == STEPS:
            val_mse, val_cos = latent_mse_cos(val_labels, model)
            checkpoint_history.append(CheckpointRecord(step=step, val_mse=val_mse))
            curves.append(
                {
                    "step": step,
                    "train_mse": sum(window_mse) / len(window_mse),
                    "train_cos": sum(window_cos) / len(window_cos),
                    "val_mse": val_mse,
                    "val_cos": val_cos,
                }
            )
            window_mse, window_cos = [], []
            if select_best_checkpoint(checkpoint_history).step == step:
                _save_atomic(run_root / "checkpoints" / "best.pt", {"step": step, "model_state_dict": model.state_dict(), "val_mse": val_mse, "val_cos": val_cos})
            if step % LOG_EVERY == 0 or step == STEPS:
                print(f"  step {step}/{STEPS}: val_mse={val_mse:.6f} val_cos={val_cos:.4f}")

    train_elapsed = time.perf_counter() - train_start
    best_record = select_best_checkpoint(checkpoint_history)
    print(f"training time: {train_elapsed:.2f}s; best checkpoint: step {best_record.step}, val_mse={best_record.val_mse:.6f}")

    best_state = torch.load(run_root / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    best_model = TemporalConvNet(channels=channels).to(device)
    best_model.load_state_dict(best_state["model_state_dict"])
    best_model.eval()

    train_mse, train_cos = latent_mse_cos(train_labels, best_model)

    mse_loss_fn_eval = nn.MSELoss()
    stft_loss_fn = MultiResolutionSTFTLoss(STFT_FFT_SIZES, STFT_HOP_SIZES, STFT_WIN_LENGTHS).to(device)
    all_labels = [s.label for s in by_split["train"] + by_split["val"] + by_split["test"]]
    shuffle_pairing = {label: all_labels[(i + 1) % len(all_labels)] for i, label in enumerate(all_labels)}

    split_metrics: dict[str, dict[str, float]] = {}
    eval_start = time.perf_counter()
    for split in ("val", "test"):
        split_specs = by_split[split]
        agg = {k: [] for k in ["mse_correct", "cos_correct", "cos_shuffled", "cos_shifted", "wav_corr_correct_vs_oracle", "stft_correct_vs_oracle"]}
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

            agg["mse_correct"].append(mse_loss_fn_eval(pred_correct, target).item())
            agg["cos_correct"].append(nn.functional.cosine_similarity(pred_correct, target, dim=-1).mean().item())
            agg["cos_shuffled"].append(nn.functional.cosine_similarity(pred_shuffled, target, dim=-1).mean().item())
            agg["cos_shifted"].append(nn.functional.cosine_similarity(pred_shifted, target, dim=-1).mean().item())

            oracle_audio = dac.decode(target.transpose(1, 2)).squeeze(0).squeeze(0)
            correct_audio = dac.decode(pred_correct.transpose(1, 2)).squeeze(0).squeeze(0)
            n = min(len(oracle_audio), len(correct_audio))
            oracle_n, correct_n = oracle_audio[:n].cpu(), correct_audio[:n].cpu()
            agg["wav_corr_correct_vs_oracle"].append(waveform_correlation(correct_n, oracle_n))
            with torch.no_grad():
                agg["stft_correct_vs_oracle"].append(
                    stft_loss_fn(correct_n.unsqueeze(0).to(device), oracle_n.unsqueeze(0).to(device))[0].item()
                )
        split_metrics[split] = {k: float(np.nanmean(v)) for k, v in agg.items()}
        split_metrics[split]["cos_margin_shuffled"] = split_metrics[split]["cos_correct"] - split_metrics[split]["cos_shuffled"]
        split_metrics[split]["cos_margin_shifted"] = split_metrics[split]["cos_correct"] - split_metrics[split]["cos_shifted"]
    eval_elapsed = time.perf_counter() - eval_start

    peak_mem_mb = torch.mps.driver_allocated_memory() / 1e6 if device.type == "mps" else None

    return {
        "layer_label": layer_label,
        "layer_index": layer_idx,
        "parameter_count": param_count,
        "best_checkpoint_step": best_record.step,
        "best_checkpoint_val_mse": best_record.val_mse,
        "train_mse": train_mse,
        "train_cos": train_cos,
        "training_time_seconds": train_elapsed,
        "evaluation_time_seconds": eval_elapsed,
        "peak_mps_driver_allocated_mb": peak_mem_mb,
        "training_curves": curves,
        "metrics_by_split": split_metrics,
    }


def main() -> None:
    torch.manual_seed(SEED)
    try:
        device = select_device(requested="auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")
    print(f"screening budget: {STEPS} steps (documented in module docstring)")
    print(f"candidate layers: {list(zip(LAYER_LABELS, CANDIDATE_INDICES, strict=True))}")

    if not LAYER_CACHE_PATH.exists():
        raise SystemExit(f"{LAYER_CACHE_PATH} not found -- run scripts/precompute_layer_sweep_cache.py first")
    if not DAC_CACHE_PATH.exists():
        raise SystemExit(f"{DAC_CACHE_PATH} not found -- this experiment reuses it from experiment/muq-to-dac-more-data")

    layer_cache = torch.load(LAYER_CACHE_PATH, map_location=device, weights_only=False)
    dac_cache = torch.load(DAC_CACHE_PATH, map_location=device, weights_only=False)
    frame_ratio = dac_cache["frame_ratio"]
    dac_targets = {label: v["dac"].to(device) for label, v in dac_cache["crops"].items()}
    print(f"loaded {len(dac_cache['crops'])} DAC targets (reused, unchanged) and {len(layer_cache['crops'])} crops x 5 MuQ layers")

    assert layer_cache["candidate_layer_indices"] == CANDIDATE_INDICES, "STOP CONDITION: cached layer indices don't match this script's selection"

    specs = load_manifest()
    by_split: dict[str, list[CropSpec]] = {"train": [], "val": [], "test": []}
    for spec in specs:
        by_split[spec.split].append(spec)
    for split, split_specs in by_split.items():
        print(f"  {split}: {len(split_specs)} crops, {len({s.track_id for s in split_specs})} tracks")

    dac = FrozenDacEncoder(DAC_MODEL_ID, device)

    muq_version = layer_cache["muq_version"]

    def load_layer_latents(layer_idx: int) -> dict[str, torch.Tensor]:
        latents = {}
        for spec in specs:
            key = layer_cache_label(spec.label, layer_idx, spec.checksum_sha256, MUQ_MODEL_ID, muq_version, "torch.float32")
            latents[spec.label] = layer_cache["crops"][spec.label][key].to(device)
        return latents

    # The final-layer control run must happen FIRST and pass its sanity check before any other
    # layer is trained, per this experiment's explicit "stop and diagnose before comparing
    # layers" requirement -- not after, as a simple loop over LAYER_LABELS would do.
    all_results: dict[str, dict] = {}
    final_idx = CANDIDATE_INDICES[LAYER_LABELS.index("final")]
    print("\n### Final-layer control run (must pass sanity check before any other layer is trained) ###")
    final_result = train_one_layer("final", final_idx, load_layer_latents(final_idx), dac_targets, frame_ratio, by_split, device, dac)
    all_results["final"] = final_result

    val_mse = final_result["best_checkpoint_val_mse"]
    val_cos = final_result["metrics_by_split"]["val"]["cos_correct"]
    print(f"\n--- Final-layer control-run sanity check: val_mse={val_mse:.4f}, val_cos={val_cos:.4f} ---")
    if val_mse > FINAL_LAYER_SANITY_VAL_MSE_MAX or val_cos < FINAL_LAYER_SANITY_VAL_COS_MIN:
        raise SystemExit(
            f"STOP CONDITION: final-layer control run did not reproduce the previous "
            f"experiment's stable region (val_mse<={FINAL_LAYER_SANITY_VAL_MSE_MAX}, "
            f"val_cos>={FINAL_LAYER_SANITY_VAL_COS_MIN} expected; got val_mse={val_mse:.4f}, "
            f"val_cos={val_cos:.4f}) -- diagnose before comparing other layers."
        )
    print("  PASSED -- reproduces the prior experiment's stable region; proceeding with the sweep.")

    for layer_label, layer_idx in zip(LAYER_LABELS, CANDIDATE_INDICES, strict=True):
        if layer_label == "final":
            continue  # already trained above, as the control run
        result = train_one_layer(layer_label, layer_idx, load_layer_latents(layer_idx), dac_targets, frame_ratio, by_split, device, dac)
        all_results[layer_label] = result

    cache_size_mb = LAYER_CACHE_PATH.stat().st_size / 1e6
    report = {
        "screening_budget_steps": STEPS,
        "candidate_layers": list(zip(LAYER_LABELS, CANDIDATE_INDICES, strict=True)),
        "layer_cache_size_mb": cache_size_mb,
        "results_by_layer": all_results,
    }
    artifact_root = REPO_ROOT / "artifacts" / RUN_ID
    artifact_root.mkdir(parents=True, exist_ok=True)
    with open(artifact_root / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {artifact_root / 'report.json'}")

    print("\n=== Comparison table (ordered by MuQ depth) ===")
    header = f"{'layer':>13s} {'idx':>3s} {'best_step':>9s} {'val_mse':>8s} {'val_cos':>8s} {'test_mse':>9s} {'test_cos':>9s} {'val_margin_shuf':>16s} {'val_margin_shift':>17s} {'val_wav_corr':>13s}"
    print(header)
    for layer_label, layer_idx in zip(LAYER_LABELS, CANDIDATE_INDICES, strict=True):
        r = all_results[layer_label]
        v, t = r["metrics_by_split"]["val"], r["metrics_by_split"]["test"]
        print(
            f"{layer_label:>13s} {layer_idx:>3d} {r['best_checkpoint_step']:>9d} "
            f"{r['best_checkpoint_val_mse']:>8.4f} {v['cos_correct']:>8.4f} "
            f"{t['mse_correct']:>9.4f} {t['cos_correct']:>9.4f} "
            f"{v['cos_margin_shuffled']:>16.4f} {v['cos_margin_shifted']:>17.4f} "
            f"{v['wav_corr_correct_vs_oracle']:>13.4f}"
        )

    best_layer = min(all_results.values(), key=lambda r: r["best_checkpoint_val_mse"])
    print(f"\nBest layer by validation MSE: {best_layer['layer_label']} (index {best_layer['layer_index']})")


if __name__ == "__main__":
    main()
