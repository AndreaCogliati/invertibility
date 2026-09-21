"""tasks/TASK-08.md: TASK-07 showed DAC's `quantized_representation` collapses toward the
random-pair floor within a fraction of one DAC frame of input shift, while candidate translator
inputs (log-mel, MuQ) stay comparatively invariant. This task extends that finding through the
DAC decoder: does a large latent-space change from a tiny shift actually produce a large change
in the *reconstructed audio*, or does DAC's decoder smooth the instability back out? No MuQ, no
translator, no training -- frozen DAC encoder-decoder only, same as TASK-07.

## Method summary (full spec: tasks/TASK-08.md)

Reuses TASK-07's exact 40-crop set (same manifest, same `select_stratified_tracks` call, same
crop-start seconds) -- asserted equal to TASK-07's own reported crop list at startup, not just
assumed from calling the same deterministic function again. Shift schedule per this task's own
(slightly extended relative to TASK-07) list: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
samples, both directions for every nonzero magnitude.

For each crop and shift: encodes+decodes the canonical (shift=0) and shifted crop independently
through the full frozen DAC encoder->quantizer->decoder pipeline, producing two reconstructed
waveforms. The two reconstructions are offset by the same `shift` samples in absolute
source-time as their inputs -- `align_shifted_pair` (see `dac_shift_sensitivity_common.py`) crops
the non-overlapping edge from whichever one leads before any audio-domain metric is computed, so
the reported numbers reflect reconstruction-quality differences, not the trivial fact that the two
signals are time-offset copies of (almost) the same audio. The same alignment-then-compare
procedure is also applied directly to the *raw* (pre-DAC) waveform pair, as the required
"does DAC amplify or attenuate the instability" reference comparison.

SI-SDR is not computed: no existing implementation was found anywhere in this project
(`grep -rniE "si.sdr|si_sdr"` across `src/`, `scripts/`, `tests/` -- no hits), and this task's own
instruction is to reuse an existing implementation, not add a new dependency solely for this
metric. Stated explicitly here rather than silently omitted.

## Similarity vs. distance, for a shared-axis figure

Waveform correlation is already a similarity in [-1, 1]. The literal "Log-Mel Distance" and
"STFT Distance" metrics this task asks for are reported as raw distances in the JSON (L2 for
log-mel, the project's own `MultiResolutionSTFTLoss` for STFT). For Figure 2 and the latent-vs-
audio correlation analysis, where a comparable "higher = more similar" scale across all three
metrics is needed, two additional *derived* quantities are also computed and clearly labeled as
such: log-mel cosine similarity (not the L2 distance) and `1 / (1 + stft_distance)` -- a monotonic
transform of the same STFT distance, not an independent metric.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from muq_inversion.dac_encoder import FrozenDacEncoder
from muq_inversion.dac_shift_sensitivity_common import (
    align_shifted_pair,
    correlate_metrics,
    cosine_similarity_per_frame,
    l2_distance_per_frame,
    load_padded_crops,
    select_stratified_tracks,
    slice_shifted,
)
from muq_inversion.gate4_common import waveform_correlation
from muq_inversion.logging_utils import select_device
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.spectrogram_encoder import LogMelSpectrogramEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/gtzan_manifest_smoke.csv"
RAW_AUDIO_ROOT = REPO_ROOT / "data/raw/gtzan"
ARTIFACT_DIR = REPO_ROOT / "artifacts/task08_dac_latent_vs_audio_shift"
TASK07_REPORT_PATH = REPO_ROOT / "artifacts/task07_dac_shift_sensitivity/shift_sensitivity_report.json"

SAMPLE_RATE = 24000
DAC_MODEL_ID = "descript/dac_24khz"

TRACKS_PER_GENRE = 2
CROP_SECONDS = 2.0
CROP_SAMPLES = round(CROP_SECONDS * SAMPLE_RATE)
CROP_START_SECONDS = [5.0, 15.0]  # identical to TASK-07 -- do not change
SHIFT_MAGNITUDES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # this task's schedule, extends TASK-07's
MAX_SHIFT = max(SHIFT_MAGNITUDES)
ASYMMETRY_THRESHOLD = 0.05

STFT_FFT_SIZES = (512, 1024, 2048)
STFT_HOP_SIZES = (50, 120, 240)
STFT_WIN_LENGTHS = (240, 600, 1200)

LISTENING_SHIFTS = [0, 2, 8, 32, 128]
LISTENING_CROP_COUNT = 2  # deterministic selection, see select_listening_crops


def select_listening_crops(crop_ids: list[str]) -> list[str]:
    """Deterministic, reproducible pick of `LISTENING_CROP_COUNT` representative crops from the
    40 -- first alphabetically and the one at the midpoint of the sorted list, for genre spread
    without randomness (`tasks/TASK-08.md` doesn't specify a selection method beyond "the same
    tracks from TASK-07")."""
    ordered = sorted(crop_ids)
    mid = len(ordered) // 2
    return [ordered[0], ordered[mid]]


def summarize_direction_pair(pos: float, neg: float) -> dict[str, float | bool]:
    asymmetric = abs(pos - neg) > ASYMMETRY_THRESHOLD
    return {"plus": pos, "minus": neg, "mean": (pos + neg) / 2, "asymmetric": asymmetric}


def audio_metrics(aligned_a: torch.Tensor, aligned_b: torch.Tensor, logmel: LogMelSpectrogramEncoder, stft_loss_fn: MultiResolutionSTFTLoss, device: torch.device) -> dict[str, float]:
    """All required audio-domain metrics for one already-aligned (a, b) pair, 1-D tensors."""
    wav_corr = waveform_correlation(aligned_a, aligned_b)
    stft_distance = float(stft_loss_fn(aligned_b.unsqueeze(0).to(device), aligned_a.unsqueeze(0).to(device))[0].item())
    mel_a = logmel.encode(aligned_a.unsqueeze(0).to(device))
    mel_b = logmel.encode(aligned_b.unsqueeze(0).to(device))
    logmel_l2 = l2_distance_per_frame(mel_a, mel_b, frame_dim=1)
    logmel_cosine = cosine_similarity_per_frame(mel_a, mel_b, frame_dim=1)
    return {
        "waveform_correlation": wav_corr,
        "stft_distance": stft_distance,
        "stft_similarity": 1.0 / (1.0 + stft_distance),  # derived, for shared-axis plotting only
        "logmel_l2_distance": logmel_l2,
        "logmel_cosine_similarity": logmel_cosine,
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "listening").mkdir(parents=True, exist_ok=True)
    device = select_device("auto", allow_cpu_fallback=False)
    print(f"device: {device}")
    t0 = time.time()

    tracks = select_stratified_tracks(MANIFEST_PATH, split="val", tracks_per_genre=TRACKS_PER_GENRE)
    buffers, track_of_crop, dropped = load_padded_crops(
        tracks, RAW_AUDIO_ROOT, SAMPLE_RATE, CROP_START_SECONDS, CROP_SAMPLES, MAX_SHIFT
    )
    crop_ids = sorted(buffers)
    print(f"{len(crop_ids)} crops loaded ({len(dropped)} dropped: {dropped})")

    with open(TASK07_REPORT_PATH) as f:
        task07_report = json.load(f)
    task07_used = sorted(task07_report["crops"]["used"])
    if crop_ids != task07_used:
        raise ValueError(
            "STOP CONDITION: crop set does not match TASK-07's exactly -- "
            f"{len(crop_ids)} vs {len(task07_used)}, first mismatch at "
            f"{next((a, b) for a, b in zip(crop_ids, task07_used) if a != b)}"
        )
    print("crop set verified identical to TASK-07's report")

    dac = FrozenDacEncoder(DAC_MODEL_ID, device)
    logmel = LogMelSpectrogramEncoder(device)
    stft_loss_fn = MultiResolutionSTFTLoss(STFT_FFT_SIZES, STFT_HOP_SIZES, STFT_WIN_LENGTHS).to(device)

    device_buffers = {c: b.to(device) for c, b in buffers.items()}

    def batch_of(shift: int) -> torch.Tensor:
        return torch.stack([slice_shifted(device_buffers[c], shift, CROP_SAMPLES, MAX_SHIFT) for c in crop_ids])

    print("encoding + decoding canonical (shift=0) crops...")
    canonical_raw = batch_of(0)
    canonical_stages = dac.encode_stages(canonical_raw)
    canonical_audio = dac.decode(canonical_stages.quantized_representation)
    recon_len = canonical_audio.shape[-1]
    print(f"DAC reconstruction length: {recon_len} (input {CROP_SAMPLES})")

    points: list[dict] = []
    points_index: dict[tuple[str, int, str], dict] = {}
    summary_by_shift: dict[int, dict] = {}

    for k in SHIFT_MAGNITUDES:
        print(f"shift={k}...")
        for sign, shift in (("plus", k), ("minus", -k)):
            shifted_raw = batch_of(shift)
            shifted_stages = dac.encode_stages(shifted_raw)
            shifted_audio = dac.decode(shifted_stages.quantized_representation)

            for i, cid in enumerate(crop_ids):
                latent_cos = cosine_similarity_per_frame(
                    canonical_stages.quantized_representation[i : i + 1],
                    shifted_stages.quantized_representation[i : i + 1],
                    frame_dim=2,
                )
                aligned_recon_a, aligned_recon_b = align_shifted_pair(canonical_audio[i], shifted_audio[i], shift)
                recon_metrics = audio_metrics(aligned_recon_a, aligned_recon_b, logmel, stft_loss_fn, device)

                aligned_raw_a, aligned_raw_b = align_shifted_pair(canonical_raw[i], shifted_raw[i], shift)
                raw_metrics = audio_metrics(aligned_raw_a, aligned_raw_b, logmel, stft_loss_fn, device)

                point = {
                    "crop_id": cid,
                    "shift": shift,
                    "shift_magnitude": k,
                    "direction": sign,
                    "latent_cosine": latent_cos,
                    "dac_reconstruction": recon_metrics,
                    "raw_reference": raw_metrics,
                }
                points.append(point)
                points_index[(cid, k, sign)] = point

    elapsed_compute = time.time() - t0
    print(f"finished per-crop/shift computation: {elapsed_compute:.1f}s")

    # --- aggregate: mean of + and - per crop per magnitude, then mean across crops ---
    for k in SHIFT_MAGNITUDES:
        crop_means: dict[str, dict] = {}
        for cid in crop_ids:
            plus_pt = points_index[(cid, k, "plus")]
            minus_pt = points_index[(cid, k, "minus")]
            crop_means[cid] = {
                "latent_cosine": summarize_direction_pair(plus_pt["latent_cosine"], minus_pt["latent_cosine"]),
                "dac_reconstruction": {
                    key: summarize_direction_pair(plus_pt["dac_reconstruction"][key], minus_pt["dac_reconstruction"][key])
                    for key in plus_pt["dac_reconstruction"]
                },
                "raw_reference": {
                    key: summarize_direction_pair(plus_pt["raw_reference"][key], minus_pt["raw_reference"][key])
                    for key in plus_pt["raw_reference"]
                },
            }
        summary_by_shift[k] = {
            "latent_cosine_mean": float(np.mean([v["latent_cosine"]["mean"] for v in crop_means.values()])),
            "dac_reconstruction_mean": {
                key: float(np.mean([v["dac_reconstruction"][key]["mean"] for v in crop_means.values()]))
                for key in next(iter(crop_means.values()))["dac_reconstruction"]
            },
            "raw_reference_mean": {
                key: float(np.mean([v["raw_reference"][key]["mean"] for v in crop_means.values()]))
                for key in next(iter(crop_means.values()))["raw_reference"]
            },
            "per_crop": crop_means,
        }

    # shift=0: trivial identity point per crop, included in the latent-vs-audio scatter for completeness.
    zero_recon_metrics = {
        "waveform_correlation": 1.0,
        "stft_distance": 0.0,
        "stft_similarity": 1.0,
        "logmel_l2_distance": 0.0,
        "logmel_cosine_similarity": 1.0,
    }
    for cid in crop_ids:
        points.append(
            {
                "crop_id": cid,
                "shift": 0,
                "shift_magnitude": 0,
                "direction": "none",
                "latent_cosine": 1.0,
                "dac_reconstruction": dict(zero_recon_metrics),
                "raw_reference": dict(zero_recon_metrics),
            }
        )
    summary_by_shift[0] = {
        "latent_cosine_mean": 1.0,
        "dac_reconstruction_mean": dict(zero_recon_metrics),
        "raw_reference_mean": dict(zero_recon_metrics),
        "per_crop": {
            cid: {
                "latent_cosine": {"plus": 1.0, "minus": 1.0, "mean": 1.0, "asymmetric": False},
                "dac_reconstruction": {k: {"plus": v, "minus": v, "mean": v, "asymmetric": False} for k, v in zero_recon_metrics.items()},
                "raw_reference": {k: {"plus": v, "minus": v, "mean": v, "asymmetric": False} for k, v in zero_recon_metrics.items()},
            }
            for cid in crop_ids
        },
    }

    # --- correlation analysis: one point per (crop, shift_magnitude) -- mean of +/- per point ---
    latent_series: list[float] = []
    audio_series: dict[str, list[float]] = {"waveform_correlation": [], "logmel_cosine_similarity": [], "stft_similarity": []}
    for k in sorted(summary_by_shift):
        for crop_summary in summary_by_shift[k]["per_crop"].values():
            latent_series.append(crop_summary["latent_cosine"]["mean"])
            for metric, values in audio_series.items():
                values.append(crop_summary["dac_reconstruction"][metric]["mean"])

    correlation_analysis = {metric: correlate_metrics(latent_series, values) for metric, values in audio_series.items()}

    report = {
        "sample_rate": SAMPLE_RATE,
        "crop_samples": CROP_SAMPLES,
        "recon_sample_length": recon_len,
        "shift_schedule": [0, *SHIFT_MAGNITUDES],
        "asymmetry_threshold": ASYMMETRY_THRESHOLD,
        "si_sdr": "not computed -- no existing implementation found in this project; "
        "task instruction is to reuse one, not add a new dependency for this metric alone",
        "crops": {"used": crop_ids, "dropped": dropped, "track_of_crop": track_of_crop},
        "summary_by_shift": summary_by_shift,
        "correlation_analysis": correlation_analysis,
        "n_latent_vs_audio_points": len(latent_series),
        "elapsed_seconds_compute": elapsed_compute,
    }
    out_path = ARTIFACT_DIR / "latent_vs_audio_shift_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")

    plot_figures(summary_by_shift, latent_series, audio_series, task07_report)
    export_listening_package(crop_ids, device_buffers, dac)

    total_elapsed = time.time() - t0
    print(f"finished: {total_elapsed:.1f}s total")


def plot_figures(summary_by_shift: dict, latent_series: list[float], audio_series: dict[str, list[float]], task07_report: dict) -> None:
    shifts = sorted(summary_by_shift)

    # --- Figure 1: latent cosine similarity vs shift (this task's own computation, same
    # methodology as TASK-07, extended to 512/1024) ---
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    latent_curve = [summary_by_shift[k]["latent_cosine_mean"] for k in shifts]
    ax1.plot(shifts[1:], latent_curve[1:], "o-", color="tab:red", label="DAC quantized_representation")
    ax1.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Shift magnitude (samples, log2 scale)")
    ax1.set_ylabel("Mean latent cosine similarity to unshifted")
    ax1.set_title("Figure 1: DAC latent similarity vs. shift (extends TASK-07 to 512/1024 samples)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(ARTIFACT_DIR / "figure1_latent_similarity_vs_shift.png", dpi=150)
    print(f"wrote {ARTIFACT_DIR / 'figure1_latent_similarity_vs_shift.png'}")

    # --- Figure 2: audio similarity vs shift, DAC reconstruction vs. raw reference ---
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    for metric, color, marker in [
        ("waveform_correlation", "tab:blue", "o"),
        ("logmel_cosine_similarity", "tab:green", "s"),
        ("stft_similarity", "tab:purple", "^"),
    ]:
        recon_curve = [summary_by_shift[k]["dac_reconstruction_mean"][metric] for k in shifts]
        raw_curve = [summary_by_shift[k]["raw_reference_mean"][metric] for k in shifts]
        ax2.plot(shifts[1:], recon_curve[1:], marker + "-", color=color, label=f"DAC reconstruction: {metric}")
        ax2.plot(shifts[1:], raw_curve[1:], marker + "--", color=color, alpha=0.5, label=f"raw reference: {metric}")
    ax2.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Shift magnitude (samples, log2 scale)")
    ax2.set_ylabel("Audio similarity (higher = more similar; stft_similarity is a derived transform)")
    ax2.set_title("Figure 2: reconstructed-audio similarity vs. shift, DAC vs. raw-waveform reference")
    ax2.legend(fontsize=7, loc="lower left")
    fig2.tight_layout()
    fig2.savefig(ARTIFACT_DIR / "figure2_audio_similarity_vs_shift.png", dpi=150)
    print(f"wrote {ARTIFACT_DIR / 'figure2_audio_similarity_vs_shift.png'}")

    # --- Figure 3: latent similarity vs audio similarity, one point per crop/shift pair ---
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    ax3.scatter(latent_series, audio_series["waveform_correlation"], c=range(len(latent_series)), cmap="viridis", s=12, alpha=0.6)
    ax3.set_xlabel("DAC latent cosine similarity")
    ax3.set_ylabel("DAC reconstruction waveform correlation")
    ax3.set_title("Figure 3: does latent similarity predict reconstruction similarity?\n(one point per crop/shift pair)")
    fig3.tight_layout()
    fig3.savefig(ARTIFACT_DIR / "figure3_latent_vs_audio_similarity.png", dpi=150)
    print(f"wrote {ARTIFACT_DIR / 'figure3_latent_vs_audio_similarity.png'}")


def export_listening_package(crop_ids: list[str], device_buffers: dict[str, torch.Tensor], dac: FrozenDacEncoder) -> None:
    listening_dir = ARTIFACT_DIR / "listening"
    listening_crops = select_listening_crops(crop_ids)
    print(f"listening crops: {listening_crops}")

    for cid in listening_crops:
        buf = device_buffers[cid]
        canonical_raw = slice_shifted(buf, 0, CROP_SAMPLES, MAX_SHIFT).unsqueeze(0)
        canonical_stages = dac.encode_stages(canonical_raw)
        canonical_audio = dac.decode(canonical_stages.quantized_representation)

        sf.write(str(listening_dir / f"{cid}_original.wav"), canonical_raw.squeeze(0).cpu().numpy().astype(np.float32), SAMPLE_RATE)
        sf.write(str(listening_dir / f"{cid}_dac_recon_shift0.wav"), canonical_audio.squeeze(0).cpu().numpy().astype(np.float32), SAMPLE_RATE)

        for shift in LISTENING_SHIFTS:
            if shift == 0:
                continue  # identical to the canonical files above -- see module docstring
            shifted_raw = slice_shifted(buf, shift, CROP_SAMPLES, MAX_SHIFT).unsqueeze(0)
            shifted_stages = dac.encode_stages(shifted_raw)
            shifted_audio = dac.decode(shifted_stages.quantized_representation)

            sf.write(str(listening_dir / f"{cid}_shift{shift}_shifted.wav"), shifted_raw.squeeze(0).cpu().numpy().astype(np.float32), SAMPLE_RATE)
            sf.write(str(listening_dir / f"{cid}_shift{shift}_dac_recon.wav"), shifted_audio.squeeze(0).cpu().numpy().astype(np.float32), SAMPLE_RATE)

            aligned_a, aligned_b = align_shifted_pair(canonical_audio.squeeze(0), shifted_audio.squeeze(0), shift)
            difference = (aligned_a - aligned_b).cpu().numpy().astype(np.float32)
            sf.write(str(listening_dir / f"{cid}_shift{shift}_dac_recon_difference.wav"), difference, SAMPLE_RATE)

    print(f"wrote listening package to {listening_dir}")


if __name__ == "__main__":
    main()
