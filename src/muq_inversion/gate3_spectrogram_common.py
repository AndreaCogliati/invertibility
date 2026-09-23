"""Pure, testable helpers for the spectrogram Gate-3 memorization control (`tasks/TASK-04.md`,
`scripts/train_spectrogram_gate3_wandb.py`) and its scaling-curve follow-up (`tasks/TASK-05.md`,
`scripts/train_gate3_scale_wandb.py`).

Kept separate from the training scripts, mirroring this project's existing convention
(`gate4_common.py`, `layer_sweep_common.py`) -- so the metric/gradient/payload logic can be
unit-tested without loading any model, cache, or W&B connection.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch import nn

from muq_inversion.dac_encoder import FrozenDacEncoder
from muq_inversion.gate4_common import waveform_correlation
from muq_inversion.latent_diagnostics import LatentStats, compute_latent_stats, ratio
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.models.spectrogram_translator import SpectrogramToDacTranslator

REPO_ROOT = Path(__file__).resolve().parents[2]
DENSE_CACHE_PATH = REPO_ROOT / "data/cache/dense/mel_dac_cache.pt"  # reused, read-only
MANIFEST_PATH = REPO_ROOT / "data/manifests/gate3_spectrogram_two_excerpt.csv"

TRACKS = ["pop.00012", "disco.00049"]
LABELS = {"pop.00012": "train_pop.00012_t14.0s", "disco.00049": "train_disco.00049_t14.0s"}


@dataclasses.dataclass(frozen=True)
class ExampleData:
    track_id: str
    mel: torch.Tensor
    dac: torch.Tensor


def _cache_label(split: str, track_id: str, crop_start_seconds: float) -> str:
    return f"{split}_{track_id}_t{crop_start_seconds:.1f}s"


def load_examples_from_manifest(manifest_path: Path, device: torch.device) -> dict[str, ExampleData]:
    """Loads an arbitrary set of examples (keyed by track_id) directly from TASK-02's dense
    cache, given a manifest CSV with the standard `split,track_id,...,crop_start_seconds,...`
    schema (`dense_crops.csv`, `gate3_scale_*.csv`, or this task's own two-excerpt manifest).
    No new cache is built -- every crop this project's Gate-3 experiments use is already present
    in `data/cache/dense/mel_dac_cache.pt` (verified at manifest-build time, not assumed here)."""
    if not DENSE_CACHE_PATH.exists():
        raise SystemExit(f"{DENSE_CACHE_PATH} not found -- reuses TASK-02's dense cache")
    cache = torch.load(DENSE_CACHE_PATH, map_location=device, weights_only=False)
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    examples: dict[str, ExampleData] = {}
    for row in rows:
        track_id = row["track_id"]
        label = _cache_label(row["split"], track_id, float(row["crop_start_seconds"]))
        if label not in cache["crops"]:
            raise SystemExit(f"STOP CONDITION: {label} not found in {DENSE_CACHE_PATH} (manifest: {manifest_path})")
        entry = cache["crops"][label]
        examples[track_id] = ExampleData(track_id=track_id, mel=entry["mel"].to(device), dac=entry["dac"].to(device))
    return examples


def load_examples(device: torch.device) -> dict[str, ExampleData]:
    """Loads TASK-04's exact two fixed excerpts. Thin wrapper over
    `load_examples_from_manifest` for backward compatibility with TASK-04's own code/tests."""
    return load_examples_from_manifest(MANIFEST_PATH, device)


def three_way_grad_norms(model: SpectrogramToDacTranslator) -> dict[str, float]:
    """mel_projection / translator / output_layer gradient-norm breakdown, per this task's exact
    metric names -- finer than `grad_diagnostics.py`'s generic top-level `named_children()`
    split, since it needs to know `TemporalConvNet`'s internal structure (conv1/conv2 = the
    residual body = "translator", conv3 = the final unrestricted-output layer =
    "output_layer"). Returns 0.0 for any group with no gradient yet (e.g. before `backward()`)."""

    def l2(params) -> float:
        grads = [p.grad.detach() for p in params if p.grad is not None]
        return float(sum(g.norm(2) ** 2 for g in grads) ** 0.5) if grads else 0.0

    tc = model.temporal_conv
    return {
        "mel_projection_l2_norm": l2(model.input_projection.parameters()),
        "translator_l2_norm": l2(list(tc.conv1.parameters()) + list(tc.conv2.parameters())),
        "output_layer_l2_norm": l2(tc.conv3.parameters()),
    }


def count_nonfinite(tensors: list[torch.Tensor]) -> int:
    return sum(int((~torch.isfinite(t)).sum().item()) for t in tensors)


def evaluate_prediction(
    pred: torch.Tensor,
    target: torch.Tensor,
    dac: FrozenDacEncoder | None,
    stft_loss_fn: MultiResolutionSTFTLoss | None,
    device: torch.device,
    compute_audio_metrics: bool,
) -> dict[str, float]:
    """Per-example memorization metrics, per tasks/TASK-04.md's exact list. Reuses
    `latent_diagnostics.py`'s and `gate4_common.py`'s implementations -- no metric redefinition.
    Takes `pred`/`target` tensors directly (not a model + example) so it works identically for
    this task's own `SpectrogramToDacTranslator` and the MuQ Gate-3 comparison's `TemporalConvNet`
    predictions, without a model-interface adapter. `dac`/`stft_loss_fn` may be `None` when
    `compute_audio_metrics=False` (they are never dereferenced in that case)."""
    pred_np = pred.squeeze(0).cpu().numpy()
    oracle_np = target.squeeze(0).cpu().numpy()

    pred_stats: LatentStats = compute_latent_stats(pred_np)
    oracle_stats: LatentStats = compute_latent_stats(oracle_np)
    ratios = ratio(pred_stats, oracle_stats)

    mse = float(nn.functional.mse_loss(pred, target).item())
    cos = float(nn.functional.cosine_similarity(pred, target, dim=-1).mean().item())
    frame_diff_mse = float(np.mean((np.diff(pred_np, axis=0) - np.diff(oracle_np, axis=0)) ** 2))

    metrics = {
        "mse": mse,
        "cosine": cos,
        "norm_ratio": ratios["norm_ratio"],
        "variance_ratio": ratios["variance_ratio"],
        "lag1_autocorrelation_prediction": pred_stats.lag1_autocorr,
        "lag1_autocorrelation_oracle": oracle_stats.lag1_autocorr,
        "frame_difference_mse": frame_diff_mse,
        "prediction_mean": float(pred_np.mean()),
        "prediction_std": float(pred_np.std()),
        "target_mean": float(oracle_np.mean()),
        "target_std": float(oracle_np.std()),
    }
    if compute_audio_metrics:
        with torch.no_grad():
            oracle_audio = dac.decode(target.transpose(1, 2)).squeeze(0).squeeze(0)
            pred_audio = dac.decode(pred.transpose(1, 2)).squeeze(0).squeeze(0)
        n = min(len(oracle_audio), len(pred_audio))
        oracle_n, pred_n = oracle_audio[:n].cpu(), pred_audio[:n].cpu()
        metrics["waveform_correlation_to_oracle"] = waveform_correlation(pred_n, oracle_n)
        with torch.no_grad():
            metrics["stft_distance_to_oracle"] = stft_loss_fn(pred_n.unsqueeze(0).to(device), oracle_n.unsqueeze(0).to(device))[0].item()
    return metrics


def build_memorization_log_payload(
    per_example: dict[str, dict[str, float]], mean_metrics: dict[str, float], tracks: list[str]
) -> dict[str, float]:
    """Flattens per-example + mean metrics into the exact `memorization/example_{i}/{metric}` and
    `memorization/mean_{metric}` W&B key structure tasks/TASK-04.md specifies. Pure function, no
    `wandb` dependency, so it's testable without a live run."""
    payload: dict[str, float] = {}
    for i, track_id in enumerate(tracks):
        for k, v in per_example[track_id].items():
            payload[f"memorization/example_{i}/{k}"] = v
    for k, v in mean_metrics.items():
        payload[f"memorization/mean_{k}"] = v
    return payload


def aggregate_metrics_across_examples(per_example: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """`tasks/TASK-05.md`'s "mean, median, worst-case across examples" requirement. Reports
    mean/median/min/max per metric rather than picking a single "worst" direction per metric
    (which is worse -- min or max -- differs by metric: higher is worse for `mse`, lower is worse
    for `cosine`), so callers/readers can interpret min or max as "worst" per their own metric.
    Raises if `per_example` is empty or examples don't share the same metric keys, so a partial
    per-example dict (e.g. one example's decode-based metrics missing) fails loudly rather than
    silently aggregating over a mismatched subset."""
    if not per_example:
        raise ValueError("per_example must be non-empty")
    metric_names = set(next(iter(per_example.values())).keys())
    for track_id, metrics in per_example.items():
        if set(metrics.keys()) != metric_names:
            raise ValueError(f"{track_id} has metric keys {set(metrics.keys())}, expected {metric_names}")

    aggregated: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = np.array([per_example[t][metric] for t in per_example])
        aggregated[metric] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return aggregated


def rank_examples_by_metric(per_example: dict[str, dict[str, float]], key: str = "waveform_correlation_to_oracle") -> list[str]:
    """`tasks/TASK-06.md`'s "identify best/median/worst example" requirement. Returns track IDs
    sorted by `key`, ascending (worst first, best last) -- the caller picks `ranking[0]` (worst),
    `ranking[len//2]` (median), `ranking[-1]` (best)."""
    return sorted(per_example.keys(), key=lambda t: per_example[t][key])
