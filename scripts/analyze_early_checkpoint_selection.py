"""TASK-03's "Additional Investigation": why does the best-by-validation-MSE checkpoint
consistently land extremely early in training?

Read-only diagnostic over already-completed runs (TASK-01, TASK-02) -- no new training. Tests
four candidate mechanisms the task asks about directly, using existing data plus one cheap
computation (a trivial "predict the dataset's mean DAC vector" baseline) rather than speculating:

1. **Optimization instability**: would show as the run never reaching a low, stable region at
   all. Ruled out as the primary explanation -- both runs do reach stable low-MSE regions many
   times after step 1000 (see the turbulence-then-recovery pattern in
   notes/DECISIONS.md#dense-spectrogram-positive-control), just never *strictly below* step
   1000's value.
2. **Validation-noise fluctuation** (pure measurement noise, no structural cause): would predict
   no systematic relationship between the selected checkpoint's MSE and any principled baseline.
   Ruled out by finding #2 below -- the match to a trivial constant-prediction baseline is too
   close and too consistent across two independent runs to be noise.
3. **Regression-to-the-mean** (the model reaches an MSE-cheap, low-information solution early,
   and validation-MSE-only selection can never find anything strictly better because leaving that
   solution costs MSE even when it gains genuine signal): supported by findings #1-#3 together.
4. **Over-smoothing** (already documented separately, e.g. lag-1 autocorrelation ~0.95-0.97 in
   [[spectrogram-positive-control]]/[[dense-spectrogram-positive-control]]): a consequence of
   mechanism #3 (a near-constant prediction is maximally "smooth"), not a distinct competing
   mechanism -- this script's finding #3 (MSE/cosine anti-correlation) is the direct evidence
   connecting the two.

Findings and full write-up: notes/DECISIONS.md#task03-bluehive-wandb.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "task01_sparse": {
        "report": REPO_ROOT / "artifacts/spectrogram_control/report.json",
        "cache": REPO_ROOT / "data/cache/more_data/muq_dac_cache.pt",  # TASK-01 reused this DAC cache
        "manifest": REPO_ROOT / "data/manifests/more_data_crops.csv",
    },
    "task02_dense": {
        "report": REPO_ROOT / "artifacts/dense_spectrogram_control/report.json",
        "cache": REPO_ROOT / "data/cache/dense/mel_dac_cache.pt",
        "manifest": REPO_ROOT / "data/manifests/dense_crops.csv",
    },
}


def load_curve(report_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(report_path) as f:
        report = json.load(f)
    curves = report["training_curves"]
    steps = np.array([c["step"] for c in curves])
    val_mse = np.array([c["val_mse"] for c in curves])
    val_cos = np.array([c["val_cos"] for c in curves])
    return steps, val_mse, val_cos


def trivial_mean_baseline_mse(cache_path: Path, manifest_path: Path) -> float:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    train_labels = [f"train_{r['track_id']}_t{float(r['crop_start_seconds']):.1f}s" for r in rows if r["split"] == "train"]
    val_labels = [f"val_{r['track_id']}_t{float(r['crop_start_seconds']):.1f}s" for r in rows if r["split"] == "val"]

    sample = train_labels[::20] if len(train_labels) > 2000 else train_labels  # subsample large sets for speed
    all_frames = torch.cat([cache["crops"][label]["dac"].squeeze(0) for label in sample], dim=0)
    mean_vec = all_frames.mean(dim=0)

    mse_loss = nn.MSELoss()
    total, n = 0.0, 0
    for label in val_labels:
        target = cache["crops"][label]["dac"].squeeze(0)
        pred = mean_vec.unsqueeze(0).expand_as(target)
        total += mse_loss(pred, target).item()
        n += 1
    return total / n


def analyze_run(name: str, cfg: dict) -> dict:
    steps, val_mse, val_cos = load_curve(cfg["report"])
    first_step_mse = val_mse[0]
    rank_of_first = int((val_mse <= first_step_mse).sum())  # 1 = global minimum

    stable_mask = val_mse < 10 * np.median(val_mse)  # exclude the huge turbulence spikes
    corr_mse_cos_all = float(np.corrcoef(val_mse, val_cos)[0, 1])
    corr_mse_cos_stable = float(np.corrcoef(val_mse[stable_mask], val_cos[stable_mask])[0, 1])

    baseline_mse = trivial_mean_baseline_mse(cfg["cache"], cfg["manifest"])
    ratio_to_baseline = float(first_step_mse / baseline_mse)

    best_cos_idx = int(val_cos.argmax())

    result = {
        "num_checkpoints": len(steps),
        "first_checkpoint_step": int(steps[0]),
        "first_checkpoint_val_mse": float(first_step_mse),
        "first_checkpoint_val_cos": float(val_cos[0]),
        "first_checkpoint_rank_by_val_mse": rank_of_first,  # 1 = never beaten for the rest of training
        "trivial_mean_vector_baseline_val_mse": baseline_mse,
        "first_checkpoint_mse_over_baseline_mse_ratio": ratio_to_baseline,
        "corr_val_mse_vs_val_cos_all_checkpoints": corr_mse_cos_all,
        "corr_val_mse_vs_val_cos_excluding_instability_spikes": corr_mse_cos_stable,
        "best_val_cos_step": int(steps[best_cos_idx]),
        "best_val_cos_value": float(val_cos[best_cos_idx]),
        "val_mse_at_best_cos_step": float(val_mse[best_cos_idx]),
    }
    print(f"\n=== {name} ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


def main() -> None:
    results = {name: analyze_run(name, cfg) for name, cfg in RUNS.items()}

    print("\n=== Interpretation ===")
    for name, r in results.items():
        print(
            f"{name}: first validation checkpoint (step {r['first_checkpoint_step']}) is rank "
            f"{r['first_checkpoint_rank_by_val_mse']} of {r['num_checkpoints']} by val MSE "
            f"(1 = global minimum, never beaten again) -- {r['first_checkpoint_mse_over_baseline_mse_ratio']:.3f}x "
            f"the trivial constant-mean-vector baseline MSE. MSE-vs-cosine correlation (excluding "
            f"instability spikes): {r['corr_val_mse_vs_val_cos_excluding_instability_spikes']:.3f} "
            f"(negative = genuine trade-off, not just noise)."
        )

    out_path = REPO_ROOT / "artifacts/task03_wandb_repro/early_checkpoint_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
