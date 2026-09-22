"""tasks/TASK-10.md: does DAC's latent-collapses-but-audio-stays-stable phenomenon (TASK-07/08)
generalize to neural audio codecs in general, or is it DAC-specific? Repeats the equivalent of
TASK-07+TASK-08 using EnCodec (`facebook/encodec_24khz`) on the exact same 40-crop dataset and
shift schedule, then builds direct DAC-vs-EnCodec comparison tables from the two codecs' already-
computed results. No training, no translator, no MuQ prediction -- pure frozen-model analysis.

## EnCodec checkpoint and measured geometry (documented per this task's explicit requirement)

- Model: `facebook/encodec_24khz` -- the same checkpoint this project used on its separate
  `encodec-hifigan-experiment` branch (see `notes/EXPERIMENT_SYNTHESIS.md`).
- Sample rate: 24000 Hz. Frame rate: 75 Hz (measured via `config.frame_rate` -- identical to
  DAC's own measured 75 Hz, so no resampling/repeat-interleave is needed to align the two codecs'
  latents frame-for-frame). Codebook count: 32 (measured via
  `EncodecResidualVectorQuantizer.get_num_quantizers_for_bandwidth(24.0)` -- identical to DAC's
  32). Latent dimensionality: 128 for both the continuous (pre-quantizer) and quantized
  (post-RVQ) representations -- notably smaller than DAC's 1024-dim `quantized_representation`.
  Full detail and verification in `src/muq_inversion/encodec_encoder.py`'s module docstring.
- **Documented EnCodec-specific preprocessing difference**: EnCodec's decoder reproduces the
  exact input length (48000 in -> 48000 out, measured), unlike DAC's own measured 48000 -> 47992
  (8 samples shorter). `align_shifted_pair` (reused unchanged from TASK-08) handles either case
  correctly since it only assumes the two signals being aligned share a common length with each
  other, not any particular relationship to the input length.
- Representation used as "the EnCodec latent" throughout: the quantized (post-RVQ) representation
  (`FrozenEncodecEncoder.encode()`), the closest analog to DAC's `quantized_representation` and
  what `EncodecModel.decode()` actually consumes -- matching TASK-07/08/09's DAC convention
  exactly. The continuous pre-quantizer embedding is also computed (`encode_stages()`) and
  reported alongside, mirroring TASK-07's DAC pre-quantizer comparison. Representation C
  (per-codebook analysis) was skipped as optional and non-trivial to add without expanding scope,
  per this task's own "only if trivial" instruction.

## Method summary

Reuses TASK-07/08/09's exact 40-crop set, shift schedule (0, 1, 2, ..., 1024 samples, both
directions), and crop-extraction/alignment logic unchanged (`dac_shift_sensitivity_common.py`'s
generic helpers, which were never DAC-specific in implementation -- only in name/docstring
context). Verified identical to TASK-09's crop list at runtime. For each crop and shift, computes
(mirroring TASK-07): cosine similarity, MSE, norm ratio (via `latent_diagnostics.py`'s existing
`compute_latent_stats`/`ratio`, reused rather than reimplemented), and codebook agreement --
between EnCodec's canonical and shifted quantized latents. Then (mirroring TASK-08): decodes both
crops through the full EnCodec pipeline and computes waveform correlation, log-mel similarity, and
STFT distance on the shift-aligned reconstructions. DAC's already-computed TASK-07/08 numbers are
loaded from their existing report.json files for the comparison tables -- not recomputed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy import stats

from muq_inversion.dac_shift_sensitivity_common import (
    align_shifted_pair,
    codebook_agreement,
    correlate_metrics,
    cosine_similarity_per_frame,
    l2_distance_per_frame,
    load_padded_crops,
    select_stratified_tracks,
    slice_shifted,
)
from muq_inversion.encodec_encoder import FrozenEncodecEncoder
from muq_inversion.gate4_common import waveform_correlation
from muq_inversion.latent_diagnostics import compute_latent_stats, ratio
from muq_inversion.logging_utils import select_device
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.spectrogram_encoder import LogMelSpectrogramEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/gtzan_manifest_smoke.csv"
RAW_AUDIO_ROOT = REPO_ROOT / "data/raw/gtzan"
ARTIFACT_DIR = REPO_ROOT / "artifacts/task10_encodec_shift_sensitivity"
TASK08_REPORT_PATH = REPO_ROOT / "artifacts/task08_dac_latent_vs_audio_shift/latent_vs_audio_shift_report.json"
TASK09_REPORT_PATH = REPO_ROOT / "artifacts/task09_dac_ot_geometry/ot_geometry_report.json"

SAMPLE_RATE = 24000
ENCODEC_MODEL_ID = "facebook/encodec_24khz"

TRACKS_PER_GENRE = 2
CROP_SECONDS = 2.0
CROP_SAMPLES = round(CROP_SECONDS * SAMPLE_RATE)
CROP_START_SECONDS = [5.0, 15.0]  # identical to TASK-07/08/09 -- do not change
SHIFT_MAGNITUDES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # identical to TASK-08/09
MAX_SHIFT = max(SHIFT_MAGNITUDES)
ASYMMETRY_THRESHOLD = 0.05

STFT_FFT_SIZES = (512, 1024, 2048)
STFT_HOP_SIZES = (50, 120, 240)
STFT_WIN_LENGTHS = (240, 600, 1200)

LISTENING_SHIFTS = [0, 2, 8, 32, 128]

ZERO_LATENT_METRICS = {"cosine": 1.0, "mse": 0.0, "norm_ratio": 1.0, "codebook_stage0_agreement": 1.0, "codebook_mean_agreement": 1.0}
ZERO_AUDIO_METRICS = {"waveform_correlation": 1.0, "stft_distance": 0.0, "stft_similarity": 1.0, "logmel_l2_distance": 0.0, "logmel_cosine_similarity": 1.0}


def select_listening_crops(crop_ids: list[str]) -> list[str]:
    """Identical selection method to TASK-08's `select_listening_crops`, for direct
    comparability of the listening packages: alphabetically first and the sorted-list midpoint."""
    ordered = sorted(crop_ids)
    mid = len(ordered) // 2
    return [ordered[0], ordered[mid]]


def summarize_direction_pair(pos: float, neg: float) -> dict[str, float | bool]:
    asymmetric = abs(pos - neg) > ASYMMETRY_THRESHOLD
    return {"plus": pos, "minus": neg, "mean": (pos + neg) / 2, "asymmetric": asymmetric}


def audio_metrics(aligned_a: torch.Tensor, aligned_b: torch.Tensor, logmel: LogMelSpectrogramEncoder, stft_loss_fn: MultiResolutionSTFTLoss, device: torch.device) -> dict[str, float]:
    """Identical implementation to TASK-08's `audio_metrics` -- reproduced here rather than
    imported, since scripts in this project do not import each other and TASK-08's own script
    must not be modified."""
    wav_corr = waveform_correlation(aligned_a, aligned_b)
    stft_distance = float(stft_loss_fn(aligned_b.unsqueeze(0).to(device), aligned_a.unsqueeze(0).to(device))[0].item())
    mel_a = logmel.encode(aligned_a.unsqueeze(0).to(device))
    mel_b = logmel.encode(aligned_b.unsqueeze(0).to(device))
    return {
        "waveform_correlation": wav_corr,
        "stft_distance": stft_distance,
        "stft_similarity": 1.0 / (1.0 + stft_distance),
        "logmel_l2_distance": l2_distance_per_frame(mel_a, mel_b, frame_dim=1),
        "logmel_cosine_similarity": cosine_similarity_per_frame(mel_a, mel_b, frame_dim=1),
    }


def latent_norm_ratio(z0: torch.Tensor, zk: torch.Tensor) -> float:
    """z0, zk: (1, channels, frames). Reuses `latent_diagnostics.py`'s existing
    `compute_latent_stats`/`ratio` (project convention throughout TASK-01/03/04/05/06) rather
    than reimplementing ||zk||/||z0|| -- `tasks/TASK-10.md`'s explicit "reuse established project
    metrics wherever practical" instruction."""
    z0_np = z0.squeeze(0).transpose(0, 1).cpu().numpy()  # (frames, channels)
    zk_np = zk.squeeze(0).transpose(0, 1).cpu().numpy()
    return ratio(compute_latent_stats(zk_np), compute_latent_stats(z0_np))["norm_ratio"]


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

    with open(TASK08_REPORT_PATH) as f:
        task08_report = json.load(f)
    with open(TASK09_REPORT_PATH) as f:
        task09_report = json.load(f)
    if crop_ids != sorted(task08_report["crops"]["used"]):
        raise ValueError("STOP CONDITION: crop set does not match TASK-08/09's exactly")
    print("crop set verified identical to TASK-07/08/09's reports")

    encodec = FrozenEncodecEncoder(ENCODEC_MODEL_ID, device)
    n_codebooks = encodec.model.quantizer.get_num_quantizers_for_bandwidth(encodec.bandwidth)
    print(f"EnCodec bandwidth={encodec.bandwidth}kbps -> n_codebooks={n_codebooks}, frame_rate={encodec.model.config.frame_rate}Hz")

    logmel = LogMelSpectrogramEncoder(device)
    stft_loss_fn = MultiResolutionSTFTLoss(STFT_FFT_SIZES, STFT_HOP_SIZES, STFT_WIN_LENGTHS).to(device)

    device_buffers = {c: b.to(device) for c, b in buffers.items()}

    def batch_of(shift: int) -> torch.Tensor:
        return torch.stack([slice_shifted(device_buffers[c], shift, CROP_SAMPLES, MAX_SHIFT) for c in crop_ids])

    print("encoding + decoding canonical (shift=0) crops...")
    canonical_stages = encodec.encode_stages(batch_of(0))
    canonical_audio = encodec.decode(canonical_stages.quantized_representation)
    recon_len = canonical_audio.shape[-1]
    print(f"EnCodec reconstruction length: {recon_len} (input {CROP_SAMPLES})")

    points_index: dict[tuple[str, int, str], dict] = {}

    for k in SHIFT_MAGNITUDES:
        print(f"shift={k}...")
        for sign, shift in (("plus", k), ("minus", -k)):
            shifted_stages = encodec.encode_stages(batch_of(shift))
            shifted_audio = encodec.decode(shifted_stages.quantized_representation)

            for i, cid in enumerate(crop_ids):
                z0 = canonical_stages.quantized_representation[i : i + 1]
                zk = shifted_stages.quantized_representation[i : i + 1]
                cosine = cosine_similarity_per_frame(z0, zk, frame_dim=2)
                mse = float(torch.nn.functional.mse_loss(z0, zk).item())
                norm_ratio = latent_norm_ratio(z0, zk)
                agreement = codebook_agreement(
                    canonical_stages.audio_codes[i : i + 1], shifted_stages.audio_codes[i : i + 1]
                )

                aligned_a, aligned_b = align_shifted_pair(canonical_audio[i], shifted_audio[i], shift)
                recon_metrics = audio_metrics(aligned_a, aligned_b, logmel, stft_loss_fn, device)

                points_index[(cid, k, sign)] = {
                    "latent": {
                        "cosine": cosine,
                        "mse": mse,
                        "norm_ratio": norm_ratio,
                        "codebook_stage0_agreement": agreement["stage0_agreement"],
                        "codebook_mean_agreement": agreement["mean_agreement_all_stages"],
                    },
                    "audio": recon_metrics,
                }

    elapsed_compute = time.time() - t0
    print(f"finished per-crop/shift computation: {elapsed_compute:.1f}s")

    # --- aggregate: mean of +/- per crop per magnitude, then mean across crops ---
    summary_by_shift: dict[int, dict] = {
        0: {
            "per_crop": {
                cid: {
                    "latent": {k: {"plus": v, "minus": v, "mean": v, "asymmetric": False} for k, v in ZERO_LATENT_METRICS.items()},
                    "audio": {k: {"plus": v, "minus": v, "mean": v, "asymmetric": False} for k, v in ZERO_AUDIO_METRICS.items()},
                }
                for cid in crop_ids
            }
        }
    }
    for k in SHIFT_MAGNITUDES:
        per_crop = {}
        for cid in crop_ids:
            plus_pt = points_index[(cid, k, "plus")]
            minus_pt = points_index[(cid, k, "minus")]
            per_crop[cid] = {
                "latent": {m: summarize_direction_pair(plus_pt["latent"][m], minus_pt["latent"][m]) for m in ZERO_LATENT_METRICS},
                "audio": {m: summarize_direction_pair(plus_pt["audio"][m], minus_pt["audio"][m]) for m in ZERO_AUDIO_METRICS},
            }
        summary_by_shift[k] = {"per_crop": per_crop}

    for k in [0, *SHIFT_MAGNITUDES]:
        per_crop = summary_by_shift[k]["per_crop"]
        summary_by_shift[k]["latent_mean"] = {m: float(np.mean([per_crop[cid]["latent"][m]["mean"] for cid in crop_ids])) for m in ZERO_LATENT_METRICS}
        summary_by_shift[k]["audio_mean"] = {m: float(np.mean([per_crop[cid]["audio"][m]["mean"] for cid in crop_ids])) for m in ZERO_AUDIO_METRICS}

    # --- correlation analysis: latent cosine vs each audio metric, one point per (crop, shift) ---
    series: dict[str, list[float]] = {"latent_cosine": [], "waveform_correlation": [], "logmel_cosine_similarity": [], "stft_distance": []}
    for k in [0, *SHIFT_MAGNITUDES]:
        for cid in crop_ids:
            crop_summary = summary_by_shift[k]["per_crop"][cid]
            series["latent_cosine"].append(crop_summary["latent"]["cosine"]["mean"])
            series["waveform_correlation"].append(crop_summary["audio"]["waveform_correlation"]["mean"])
            series["logmel_cosine_similarity"].append(crop_summary["audio"]["logmel_cosine_similarity"]["mean"])
            series["stft_distance"].append(crop_summary["audio"]["stft_distance"]["mean"])

    encodec_correlation = {
        audio_metric: correlate_metrics(series["latent_cosine"], series[audio_metric])
        for audio_metric in ("waveform_correlation", "logmel_cosine_similarity", "stft_distance")
    }

    # --- DAC comparison, loaded from TASK-07/08's existing reports, not recomputed ---
    # DAC latent cosine pulled from TASK-09's report (not TASK-07's, which only covers shifts up to
    # 256): TASK-09 recomputed DAC's latent cosine with the exact same methodology as TASK-07,
    # extended to the full 0-1024 schedule this task needs.
    dac_latent_cosine_by_shift = {k: task09_report["summary_by_shift"][str(k)]["cosine_mean"] for k in SHIFT_MAGNITUDES}
    dac_audio_wavcorr_by_shift = {k: task08_report["summary_by_shift"][str(k)]["dac_reconstruction_mean"]["waveform_correlation"] for k in SHIFT_MAGNITUDES}
    dac_corr = task08_report["correlation_analysis"]
    dac_alignment = {
        "waveform_correlation": dac_corr["waveform_correlation"]["spearman_r"],
        "logmel_cosine_similarity": dac_corr["logmel_cosine_similarity"]["spearman_r"],
        # DAC's TASK-08 report correlated latent cosine against the derived stft_similarity=1/(1+distance),
        # a monotonically decreasing transform of stft_distance -- negating gives the equivalent
        # correlation against stft_distance directly, for apples-to-apples comparison with EnCodec below.
        "stft_distance": -dac_corr["stft_similarity"]["spearman_r"],
    }
    encodec_alignment = {m: encodec_correlation[m]["spearman_r"] for m in ("waveform_correlation", "logmel_cosine_similarity", "stft_distance")}

    latent_stability_table = [{"shift": k, "dac_cosine": dac_latent_cosine_by_shift[k], "encodec_cosine": summary_by_shift[k]["latent_mean"]["cosine"]} for k in SHIFT_MAGNITUDES]
    audio_stability_table = [{"shift": k, "dac_waveform_correlation": dac_audio_wavcorr_by_shift[k], "encodec_waveform_correlation": summary_by_shift[k]["audio_mean"]["waveform_correlation"]} for k in SHIFT_MAGNITUDES]

    total_elapsed = time.time() - t0
    report = {
        "encodec_model_id": ENCODEC_MODEL_ID,
        "encodec_bandwidth_kbps": encodec.bandwidth,
        "encodec_n_codebooks": n_codebooks,
        "encodec_frame_rate_hz": encodec.model.config.frame_rate,
        "encodec_recon_sample_length": recon_len,
        "sample_rate": SAMPLE_RATE,
        "crop_samples": CROP_SAMPLES,
        "shift_schedule": [0, *SHIFT_MAGNITUDES],
        "crops": {"used": crop_ids, "dropped": dropped, "track_of_crop": track_of_crop},
        "summary_by_shift": summary_by_shift,
        "encodec_correlation_vs_latent_cosine": encodec_correlation,
        "comparison": {
            "latent_stability_table": latent_stability_table,
            "audio_stability_table": audio_stability_table,
            "latent_to_audio_alignment": {"dac": dac_alignment, "encodec": encodec_alignment},
        },
        "n_points": len(series["latent_cosine"]),
        "elapsed_seconds": total_elapsed,
    }
    out_path = ARTIFACT_DIR / "encodec_shift_sensitivity_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")

    plot_figures(summary_by_shift, series, dac_latent_cosine_by_shift, dac_audio_wavcorr_by_shift)
    export_listening_package(crop_ids, device_buffers, encodec)

    print(f"finished: {total_elapsed:.1f}s total")


def plot_figures(summary_by_shift: dict, series: dict, dac_latent_cosine_by_shift: dict, dac_audio_wavcorr_by_shift: dict) -> None:
    shifts = SHIFT_MAGNITUDES

    # --- Latent similarity vs shift (EnCodec, with DAC overlay) ---
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))
    encodec_cosine = [summary_by_shift[k]["latent_mean"]["cosine"] for k in shifts]
    dac_cosine = [dac_latent_cosine_by_shift[k] for k in shifts]
    ax1.plot(shifts, encodec_cosine, "o-", color="tab:blue", label="EnCodec")
    ax1.plot(shifts, dac_cosine, "o--", color="tab:red", alpha=0.7, label="DAC (from TASK-09, extends TASK-07)")
    ax1.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Shift magnitude (samples, log2)")
    ax1.set_ylabel("Mean latent cosine similarity to unshifted")
    ax1.set_title("Latent similarity vs. shift: EnCodec vs. DAC")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(ARTIFACT_DIR / "figure1_latent_similarity_vs_shift.png", dpi=150)
    print(f"wrote {ARTIFACT_DIR / 'figure1_latent_similarity_vs_shift.png'}")

    # --- Audio similarity vs shift (EnCodec, with DAC overlay) ---
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    encodec_wavcorr = [summary_by_shift[k]["audio_mean"]["waveform_correlation"] for k in shifts]
    dac_wavcorr = [dac_audio_wavcorr_by_shift[k] for k in shifts]
    ax2.plot(shifts, encodec_wavcorr, "o-", color="tab:blue", label="EnCodec waveform correlation")
    ax2.plot(shifts, dac_wavcorr, "o--", color="tab:red", alpha=0.7, label="DAC waveform correlation (from TASK-08)")
    ax2.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Shift magnitude (samples, log2)")
    ax2.set_ylabel("Audio similarity")
    ax2.set_title("Reconstructed-audio similarity vs. shift: EnCodec vs. DAC")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(ARTIFACT_DIR / "figure2_audio_similarity_vs_shift.png", dpi=150)
    print(f"wrote {ARTIFACT_DIR / 'figure2_audio_similarity_vs_shift.png'}")

    # --- Latent similarity vs audio similarity, 3 scatter panels ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key, title in [
        (axes3[0], "waveform_correlation", "Latent cosine vs. waveform correlation"),
        (axes3[1], "logmel_cosine_similarity", "Latent cosine vs. log-mel cosine similarity"),
        (axes3[2], "stft_distance", "Latent cosine vs. STFT distance"),
    ]:
        x = np.array(series["latent_cosine"])
        y = np.array(series[key])
        ax.scatter(x, y, s=10, alpha=0.4, color="tab:blue")
        rho = stats.spearmanr(x, y).statistic
        ax.set_title(f"{title}\nSpearman rho={rho:.3f}")
        ax.set_xlabel("EnCodec latent cosine similarity")
        ax.set_ylabel(key)
    fig3.suptitle("EnCodec: latent similarity vs. audio similarity (one point per crop/shift pair)")
    fig3.tight_layout()
    fig3.savefig(ARTIFACT_DIR / "figure3_latent_vs_audio_similarity.png", dpi=150)
    print(f"wrote {ARTIFACT_DIR / 'figure3_latent_vs_audio_similarity.png'}")


def export_listening_package(crop_ids: list[str], device_buffers: dict[str, torch.Tensor], encodec: FrozenEncodecEncoder) -> None:
    listening_dir = ARTIFACT_DIR / "listening"
    listening_crops = select_listening_crops(crop_ids)
    print(f"listening crops: {listening_crops}")

    for cid in listening_crops:
        buf = device_buffers[cid]
        canonical_raw = slice_shifted(buf, 0, CROP_SAMPLES, MAX_SHIFT).unsqueeze(0)
        canonical_audio = encodec.decode(encodec.encode(canonical_raw))

        sf.write(str(listening_dir / f"{cid}_original.wav"), canonical_raw.squeeze(0).cpu().numpy().astype(np.float32), SAMPLE_RATE)
        sf.write(str(listening_dir / f"{cid}_encodec_recon_shift0.wav"), canonical_audio.squeeze(0).cpu().numpy().astype(np.float32), SAMPLE_RATE)

        for shift in LISTENING_SHIFTS:
            if shift == 0:
                continue
            shifted_raw = slice_shifted(buf, shift, CROP_SAMPLES, MAX_SHIFT).unsqueeze(0)
            shifted_audio = encodec.decode(encodec.encode(shifted_raw))

            sf.write(str(listening_dir / f"{cid}_shift{shift}_shifted.wav"), shifted_raw.squeeze(0).cpu().numpy().astype(np.float32), SAMPLE_RATE)
            sf.write(str(listening_dir / f"{cid}_shift{shift}_encodec_recon.wav"), shifted_audio.squeeze(0).cpu().numpy().astype(np.float32), SAMPLE_RATE)

    print(f"wrote listening package to {listening_dir}")


if __name__ == "__main__":
    main()
