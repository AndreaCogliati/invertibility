"""tasks/TASK-09.md: TASK-07 found DAC's `quantized_representation` collapses under tiny input
shifts; TASK-08 found the reconstructed audio barely changes over that same range. This task asks
whether the mismatch is an artifact of *pointwise* comparison (frame i vs. frame i): an Optimal
Transport (OT) distance treats each latent as an unordered cloud of 150 points in 1024 dimensions
and asks how much mass must move to turn one cloud into the other, rather than requiring frame i
to line up with frame i. No training, no checkpoints, no translator -- pure analysis.

## Method summary (full spec: tasks/TASK-09.md)

Reuses TASK-07/TASK-08's exact 40-crop set and shift schedule (0, 1, 2, ..., 1024 samples),
verified identical to TASK-08's own report at startup. Latent tensors are recomputed (TASK-08
only persisted scalar summary metrics to disk, not raw latent tensors, so there is nothing to
literally reuse for MSE/cosine/OT) -- but the reconstructed-audio similarity metrics ARE reused
verbatim from TASK-08's report.json, per this task's "reuse the exact same... reconstructed
audio" instruction: no DAC decode is run in this script at all.

For each crop and shift, three latent-space distance metrics are computed between the canonical
(shift=0) and shifted `quantized_representation`:
  - MSE (`torch.nn.functional.mse_loss`, matching every other MSE in this project);
  - cosine similarity (`cosine_similarity_per_frame`, matching TASK-07/08's implementation);
  - Optimal Transport: the debiased entropic Sinkhorn divergence (POT's
    `ot.bregman.empirical_sinkhorn_divergence`, squared-Euclidean ground cost) between the two
    latents treated as 150-point clouds in 1024 dimensions, each point uniformly weighted.

## OT implementation details (documented per this task's explicit requirement)

- **Library**: `POT` (python-optimal-transport, `pip`/`uv` package `pot`), added as a new project
  dependency (`uv add pot`) -- this task's own instruction explicitly permits either `geomloss` or
  `POT` for this purpose, unlike TASK-08's SI-SDR instruction (reuse-only, no new dependency).
  POT was chosen over geomloss for a lighter dependency footprint (no GPU-kernel compilation
  backend needed for a diagnostic run) and a more direct primitive for exactly what this task asks
  for (`empirical_sinkhorn_divergence` takes raw point clouds, not a battery of loss-layer
  abstractions).
- **Ground cost**: squared Euclidean distance between frame vectors (`metric="sqeuclidean"`),
  matching MSE's own quadratic geometry -- deliberately, so a difference between the OT and MSE
  numbers reflects the transport-vs-pointwise question this task asks, not a different notion of
  per-point distance.
- **Regularization (epsilon)**: `sinkhorn_epsilon_for_cloud` (`dac_shift_sensitivity_common.py`) --
  0.1 times the median of the nonzero pairwise squared distances *within the canonical latent's
  own point cloud*, computed once per crop and reused for every shift comparison of that crop.
  Scaling to each crop's own distance scale (rather than one fixed absolute epsilon across all 40
  crops, whose raw latent magnitudes are not assumed equal) keeps Sinkhorn numerically
  well-conditioned without per-pair tuning. `fraction=0.1` was chosen empirically (not tuned
  aggressively, per this task's instruction): a quick trial on real DAC latents at `fraction=0.05`
  failed to converge within the default iteration budget for a small shift, while `0.1` converged
  cleanly across the shifts tried; this was checked once, not swept.
- **Divergence, not raw Sinkhorn cost**: the plain entropic OT cost (`ot.sinkhorn2`) does not
  vanish for identical clouds (an O(epsilon) entropy bias even at zero true distance, confirmed
  empirically before deciding). The debiased Sinkhorn divergence subtracts the two self-transport
  terms and returns exactly 0.0 for identical inputs, giving an interpretable zero-point that
  matches MSE=0/cosine=1 at shift=0.
- **Iteration budget**: `numIterMax=5000` (default: 1000) -- widened once, for safety margin
  against non-convergence warnings seen in an early trial, not as a per-run tuning knob.

## Similarity vs. distance sign convention for the correlation analysis

MSE and OT are distances (larger = more different); cosine similarity and the two audio
"similarity" metrics are similarities (larger = more alike); STFT distance is a distance. Raw
signed Pearson/Spearman correlations are reported for every (latent metric, audio metric) pair in
the JSON. The headline comparison table (matching this task's own Outcome A/B/C example format,
which reports MSE/Cosine/OT as directly comparable positive numbers) uses `abs(correlation)`
against waveform correlation specifically, since Figure 3 and the interpretation rules are framed
around that one audio metric.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from muq_inversion.dac_encoder import FrozenDacEncoder
from muq_inversion.dac_shift_sensitivity_common import (
    cosine_similarity_per_frame,
    load_padded_crops,
    select_stratified_tracks,
    sinkhorn_epsilon_for_cloud,
    sinkhorn_ot_divergence,
    slice_shifted,
)
from muq_inversion.logging_utils import select_device

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/gtzan_manifest_smoke.csv"
RAW_AUDIO_ROOT = REPO_ROOT / "data/raw/gtzan"
ARTIFACT_DIR = REPO_ROOT / "artifacts/task09_dac_ot_geometry"
TASK08_REPORT_PATH = REPO_ROOT / "artifacts/task08_dac_latent_vs_audio_shift/latent_vs_audio_shift_report.json"

SAMPLE_RATE = 24000
DAC_MODEL_ID = "descript/dac_24khz"

TRACKS_PER_GENRE = 2
CROP_SECONDS = 2.0
CROP_SAMPLES = round(CROP_SECONDS * SAMPLE_RATE)
CROP_START_SECONDS = [5.0, 15.0]  # identical to TASK-07/08 -- do not change
SHIFT_MAGNITUDES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
MAX_SHIFT = max(SHIFT_MAGNITUDES)
ASYMMETRY_THRESHOLD = 0.05

OT_EPSILON_FRACTION = 0.1
OT_NUM_ITER_MAX = 5000

ZERO_RECON_METRICS = {
    "waveform_correlation": 1.0,
    "stft_distance": 0.0,
    "stft_similarity": 1.0,
    "logmel_l2_distance": 0.0,
    "logmel_cosine_similarity": 1.0,
}


def summarize_direction_pair(pos: float, neg: float) -> dict[str, float | bool]:
    asymmetric = abs(pos - neg) > ASYMMETRY_THRESHOLD
    return {"plus": pos, "minus": neg, "mean": (pos + neg) / 2, "asymmetric": asymmetric}


def rank_agreement(latent_values: list[float], audio_values: list[float], lower_is_more_similar: bool) -> dict[str, float]:
    """Ranks both series from most-similar to least-similar and reports how well the rankings
    agree -- `tasks/TASK-09.md`'s Figure 4 "Ranking Comparison" requirement, expressed
    numerically (the figure plots the same underlying rank pairs)."""
    latent_arr = np.asarray(latent_values)
    audio_arr = np.asarray(audio_values)
    latent_rank = stats.rankdata(latent_arr if lower_is_more_similar else -latent_arr)
    audio_rank = stats.rankdata(-audio_arr)  # audio_values is always a similarity: higher = more similar
    rho = stats.spearmanr(latent_rank, audio_rank).statistic
    return {"spearman_of_ranks": float(rho)}


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    device = select_device("auto", allow_cpu_fallback=False)
    print(f"device: {device}")
    t0 = time.time()

    tracks = select_stratified_tracks(MANIFEST_PATH, split="val", tracks_per_genre=TRACKS_PER_GENRE)
    buffers, track_of_crop, dropped = load_padded_crops(
        tracks, RAW_AUDIO_ROOT, SAMPLE_RATE, CROP_START_SECONDS, CROP_SAMPLES, MAX_SHIFT
    )
    crop_ids = sorted(buffers)
    print(f"{len(crop_ids)} crops loaded ({len(dropped)} dropped: {dropped})")

    with open(TASK08_REPORT_PATH) as f:
        task08_report = json.load(f)
    task08_used = sorted(task08_report["crops"]["used"])
    if crop_ids != task08_used:
        raise ValueError("STOP CONDITION: crop set does not match TASK-08's exactly")
    print("crop set verified identical to TASK-08's report")

    dac = FrozenDacEncoder(DAC_MODEL_ID, device)
    device_buffers = {c: b.to(device) for c, b in buffers.items()}

    def batch_of(shift: int) -> torch.Tensor:
        return torch.stack([slice_shifted(device_buffers[c], shift, CROP_SAMPLES, MAX_SHIFT) for c in crop_ids])

    print("encoding canonical (shift=0) crops...")
    canonical_latent = dac.encode(batch_of(0))  # (40, 1024, frames)
    canonical_latent_np = canonical_latent.transpose(1, 2).cpu().numpy().astype(np.float64)  # (40, frames, 1024)

    print("computing per-crop OT epsilon (from each crop's own canonical point cloud)...")
    eps_per_crop = {cid: sinkhorn_epsilon_for_cloud(canonical_latent_np[i], OT_EPSILON_FRACTION) for i, cid in enumerate(crop_ids)}

    points_index: dict[tuple[str, int, str], dict] = {}
    ot_time_total = 0.0

    for k in SHIFT_MAGNITUDES:
        print(f"shift={k}...")
        for sign, shift in (("plus", k), ("minus", -k)):
            shifted_latent = dac.encode(batch_of(shift))
            shifted_latent_np = shifted_latent.transpose(1, 2).cpu().numpy().astype(np.float64)

            for i, cid in enumerate(crop_ids):
                mse = float(torch.nn.functional.mse_loss(canonical_latent[i : i + 1], shifted_latent[i : i + 1]).item())
                cosine = cosine_similarity_per_frame(canonical_latent[i : i + 1], shifted_latent[i : i + 1], frame_dim=2)
                ot_t0 = time.time()
                ot_divergence = sinkhorn_ot_divergence(
                    canonical_latent_np[i], shifted_latent_np[i], epsilon=eps_per_crop[cid], num_iter_max=OT_NUM_ITER_MAX
                )
                ot_time_total += time.time() - ot_t0

                points_index[(cid, k, sign)] = {"mse": mse, "cosine": cosine, "ot_divergence": ot_divergence}

    elapsed_compute = time.time() - t0
    print(f"finished per-crop/shift computation: {elapsed_compute:.1f}s (OT calls: {ot_time_total:.1f}s)")

    # --- aggregate: mean of +/- per crop per magnitude ---
    summary_by_shift: dict[int, dict] = {
        0: {
            "mse_mean": 0.0,
            "cosine_mean": 1.0,
            "ot_divergence_mean": 0.0,
            "per_crop": {
                cid: {
                    "mse": {"plus": 0.0, "minus": 0.0, "mean": 0.0, "asymmetric": False},
                    "cosine": {"plus": 1.0, "minus": 1.0, "mean": 1.0, "asymmetric": False},
                    "ot_divergence": {"plus": 0.0, "minus": 0.0, "mean": 0.0, "asymmetric": False},
                }
                for cid in crop_ids
            },
        }
    }
    for k in SHIFT_MAGNITUDES:
        per_crop = {}
        for cid in crop_ids:
            plus_pt = points_index[(cid, k, "plus")]
            minus_pt = points_index[(cid, k, "minus")]
            per_crop[cid] = {
                metric: summarize_direction_pair(plus_pt[metric], minus_pt[metric]) for metric in ("mse", "cosine", "ot_divergence")
            }
        summary_by_shift[k] = {
            "mse_mean": float(np.mean([v["mse"]["mean"] for v in per_crop.values()])),
            "cosine_mean": float(np.mean([v["cosine"]["mean"] for v in per_crop.values()])),
            "ot_divergence_mean": float(np.mean([v["ot_divergence"]["mean"] for v in per_crop.values()])),
            "per_crop": per_crop,
        }

    # --- pull TASK-08's reconstructed-audio similarity metrics verbatim, no DAC decode here ---
    audio_by_shift: dict[int, dict] = {0: {cid: dict(ZERO_RECON_METRICS) for cid in crop_ids}}
    for k in SHIFT_MAGNITUDES:
        audio_by_shift[k] = {
            cid: {
                metric: task08_report["summary_by_shift"][str(k)]["per_crop"][cid]["dac_reconstruction"][metric]["mean"]
                for metric in ZERO_RECON_METRICS
            }
            for cid in crop_ids
        }

    # --- combined per-(crop, shift_magnitude) dataset, one point per pair, matching TASK-08's convention ---
    shifts_all = [0, *SHIFT_MAGNITUDES]
    series: dict[str, list[float]] = {
        "mse": [],
        "cosine": [],
        "ot_divergence": [],
        "waveform_correlation": [],
        "logmel_cosine_similarity": [],
        "stft_distance": [],
    }
    for k in shifts_all:
        for cid in crop_ids:
            crop_summary = summary_by_shift[k]["per_crop"][cid]
            series["mse"].append(crop_summary["mse"]["mean"])
            series["cosine"].append(crop_summary["cosine"]["mean"])
            series["ot_divergence"].append(crop_summary["ot_divergence"]["mean"])
            audio = audio_by_shift[k][cid]
            series["waveform_correlation"].append(audio["waveform_correlation"])
            series["logmel_cosine_similarity"].append(audio["logmel_cosine_similarity"])
            series["stft_distance"].append(audio["stft_distance"])

    def correlate(x, y):
        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        return {"pearson_r": float(pearson.statistic), "spearman_r": float(spearman.statistic)}

    correlation_table = {
        latent_metric: {audio_metric: correlate(series[latent_metric], series[audio_metric]) for audio_metric in ("waveform_correlation", "logmel_cosine_similarity", "stft_distance")}
        for latent_metric in ("mse", "cosine", "ot_divergence")
    }

    headline_vs_waveform_corr = {
        latent_metric: {
            "pearson_r": correlation_table[latent_metric]["waveform_correlation"]["pearson_r"],
            "spearman_r": correlation_table[latent_metric]["waveform_correlation"]["spearman_r"],
            "abs_spearman_r": abs(correlation_table[latent_metric]["waveform_correlation"]["spearman_r"]),
        }
        for latent_metric in ("mse", "cosine", "ot_divergence")
    }

    ranking_agreement = {
        "mse": rank_agreement(series["mse"], series["waveform_correlation"], lower_is_more_similar=True),
        "cosine": rank_agreement(series["cosine"], series["waveform_correlation"], lower_is_more_similar=False),
        "ot_divergence": rank_agreement(series["ot_divergence"], series["waveform_correlation"], lower_is_more_similar=True),
    }

    total_elapsed = time.time() - t0
    report = {
        "sample_rate": SAMPLE_RATE,
        "crop_samples": CROP_SAMPLES,
        "shift_schedule": shifts_all,
        "ot_implementation": {
            "library": "POT (python-optimal-transport), added via `uv add pot`",
            "function": "ot.bregman.empirical_sinkhorn_divergence",
            "ground_cost": "sqeuclidean",
            "epsilon_fraction": OT_EPSILON_FRACTION,
            "epsilon_definition": "fraction * median(nonzero pairwise sqeuclidean distances within each crop's own canonical point cloud), computed once per crop",
            "num_iter_max": OT_NUM_ITER_MAX,
            "point_cloud_shape": list(canonical_latent_np.shape[1:]),
        },
        "crops": {"used": crop_ids, "dropped": dropped, "track_of_crop": track_of_crop},
        "eps_per_crop": eps_per_crop,
        "summary_by_shift": summary_by_shift,
        "correlation_table": correlation_table,
        "headline_vs_waveform_correlation": headline_vs_waveform_corr,
        "ranking_agreement_vs_waveform_correlation": ranking_agreement,
        "n_points": len(series["mse"]),
        "runtime_seconds": {"total": total_elapsed, "per_crop_shift_compute": elapsed_compute, "ot_calls_only": ot_time_total},
    }
    out_path = ARTIFACT_DIR / "ot_geometry_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")

    plot_figures(summary_by_shift, audio_by_shift, series, shifts_all)

    print(f"finished: {total_elapsed:.1f}s total")
    print(f"headline (|Spearman| vs waveform correlation): {json.dumps({k: v['abs_spearman_r'] for k, v in headline_vs_waveform_corr.items()}, indent=2)}")


def plot_figures(summary_by_shift: dict, audio_by_shift: dict, series: dict, shifts_all: list[int]) -> None:
    shifts = [s for s in shifts_all if s != 0]

    # --- Figure 1: latent distance vs shift (MSE, cosine, OT) ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title in [
        (axes1[0], "mse_mean", "MSE"),
        (axes1[1], "cosine_mean", "Cosine similarity"),
        (axes1[2], "ot_divergence_mean", "OT (Sinkhorn divergence)"),
    ]:
        values = [summary_by_shift[k][key] for k in shifts]
        ax.plot(shifts, values, "o-", color="tab:red")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Shift magnitude (samples, log2)")
        ax.set_title(title)
    fig1.suptitle("Figure 1: latent distance vs. shift")
    fig1.tight_layout()
    fig1.savefig(ARTIFACT_DIR / "figure1_latent_distance_vs_shift.png", dpi=150)
    print(f"wrote {ARTIFACT_DIR / 'figure1_latent_distance_vs_shift.png'}")

    # --- Figure 2: audio similarity vs shift (reuse TASK-08 metrics) ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for metric, color in [("waveform_correlation", "tab:blue"), ("logmel_cosine_similarity", "tab:green")]:
        values = [np.mean([audio_by_shift[k][cid][metric] for cid in audio_by_shift[k]]) for k in shifts]
        ax2.plot(shifts, values, "o-", color=color, label=metric)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Shift magnitude (samples, log2)")
    ax2.set_ylabel("Audio similarity")
    ax2.set_title("Figure 2: reconstructed-audio similarity vs. shift (reused from TASK-08)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(ARTIFACT_DIR / "figure2_audio_similarity_vs_shift.png", dpi=150)
    print(f"wrote {ARTIFACT_DIR / 'figure2_audio_similarity_vs_shift.png'}")

    # --- Figure 3 (most important): distance metric vs audio similarity, 3 scatter panels ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key, title in [
        (axes3[0], "mse", "MSE vs. waveform correlation"),
        (axes3[1], "cosine", "Cosine similarity vs. waveform correlation"),
        (axes3[2], "ot_divergence", "OT divergence vs. waveform correlation"),
    ]:
        x = np.array(series[key])
        y = np.array(series["waveform_correlation"])
        ax.scatter(x, y, s=10, alpha=0.4, color="tab:purple")
        if np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept, color="black", linewidth=1)
        rho = stats.spearmanr(x, y).statistic
        ax.set_title(f"{title}\nSpearman rho={rho:.3f}")
        ax.set_xlabel(key)
        ax.set_ylabel("waveform correlation")
    fig3.suptitle("Figure 3: distance metric vs. audio similarity (one point per crop/shift pair)")
    fig3.tight_layout()
    fig3.savefig(ARTIFACT_DIR / "figure3_distance_vs_audio_similarity.png", dpi=150)
    print(f"wrote {ARTIFACT_DIR / 'figure3_distance_vs_audio_similarity.png'}")

    # --- Figure 4: ranking comparison ---
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
    audio_rank = stats.rankdata(-np.array(series["waveform_correlation"]))
    for ax, key, lower_is_more_similar, title in [
        (axes4[0], "mse", True, "MSE rank vs. audio-similarity rank"),
        (axes4[1], "cosine", False, "Cosine-similarity rank vs. audio-similarity rank"),
        (axes4[2], "ot_divergence", True, "OT-divergence rank vs. audio-similarity rank"),
    ]:
        x = np.array(series[key])
        latent_rank = stats.rankdata(x if lower_is_more_similar else -x)
        ax.scatter(latent_rank, audio_rank, s=10, alpha=0.4, color="tab:orange")
        n = len(latent_rank)
        ax.plot([1, n], [1, n], color="black", linewidth=1, linestyle="--")
        rho = stats.spearmanr(latent_rank, audio_rank).statistic
        ax.set_title(f"{title}\nrank-agreement rho={rho:.3f}")
        ax.set_xlabel(f"{key} rank (1=most similar)")
        ax.set_ylabel("audio-similarity rank (1=most similar)")
    fig4.suptitle("Figure 4: ranking comparison, latent metric vs. audio similarity")
    fig4.tight_layout()
    fig4.savefig(ARTIFACT_DIR / "figure4_ranking_comparison.png", dpi=150)
    print(f"wrote {ARTIFACT_DIR / 'figure4_ranking_comparison.png'}")


if __name__ == "__main__":
    main()
