"""tasks/TASK-11.md's four required figures, built from `scripts/train_gate3_encodec_scale_wandb.py`'s
`artifacts/task11_gate3_encodec_scale/n{2,8,32}/report.json` (best-by-MSE checkpoint per level)
and TASK-05's existing `artifacts/task05_gate3_scale/n{2,8,32}/report.json` (DAC comparison,
reused unchanged).

Read-only over the existing reports; does not retrain or touch any run/checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
ENCODEC_DIR = REPO_ROOT / "artifacts/task11_gate3_encodec_scale"
DAC_DIR = REPO_ROOT / "artifacts/task05_gate3_scale"
LEVELS = [2, 8, 32]


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_reports() -> tuple[dict, dict]:
    encodec_reports = {n: _load_json(ENCODEC_DIR / f"n{n}/report.json") for n in LEVELS}
    dac_reports = {n: _load_json(DAC_DIR / f"n{n}/report.json") for n in LEVELS}
    return encodec_reports, dac_reports


def figure1_encodec_scaling_curve(encodec_reports: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    panels = [
        ("cosine", "Latent Cosine Similarity", axes[0]),
        ("norm_ratio", "Norm Ratio", axes[1]),
        ("variance_ratio", "Variance Ratio", axes[2]),
    ]
    for key, title, ax in panels:
        values = [encodec_reports[n]["best_checkpoint"]["aggregate_latent_metrics"][key]["mean"] for n in LEVELS]
        ax.plot(LEVELS, values, "o-", color="tab:blue")
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(LEVELS)
        ax.set_xticklabels([str(n) for n in LEVELS])
        ax.set_xlabel("N (training examples)")
        ax.set_title(title)

    wav_corr = [encodec_reports[n]["best_checkpoint"]["aggregate_audio_metrics"]["waveform_correlation_to_oracle"]["mean"] for n in LEVELS]
    oracle_wav_corr = [encodec_reports[n]["oracle_aggregate"]["waveform_correlation"]["mean"] for n in LEVELS]
    axes[3].plot(LEVELS, wav_corr, "o-", color="tab:blue", label="translator (pred vs. target-decode)")
    axes[3].plot(LEVELS, oracle_wav_corr, "o--", color="tab:gray", label="EnCodec oracle (orig vs. oracle-decode)")
    axes[3].axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    axes[3].set_xscale("log", base=2)
    axes[3].set_xticks(LEVELS)
    axes[3].set_xticklabels([str(n) for n in LEVELS])
    axes[3].set_xlabel("N (training examples)")
    axes[3].set_title("Waveform Correlation")
    axes[3].legend(fontsize=7)

    fig.suptitle("Figure 1: EnCodec continuous-latent scaling curve (best-by-MSE checkpoint)")
    fig.tight_layout()
    out = ENCODEC_DIR / "figure1_encodec_scaling_curve.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def figure2_dac_vs_encodec_scaling(encodec_reports: dict, dac_reports: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("cosine", "Latent Cosine Similarity", axes[0]),
        ("waveform_correlation_to_oracle", "Waveform Correlation (pred vs. target-decode)", axes[1]),
        ("variance_ratio", "Variance Ratio", axes[2]),
    ]
    for key, title, ax in panels:
        if key == "waveform_correlation_to_oracle":
            encodec_vals = [encodec_reports[n]["best_checkpoint"]["aggregate_audio_metrics"][key]["mean"] for n in LEVELS]
            dac_vals = [dac_reports[n]["final_aggregate_fixed_subset_audio_metrics"][key]["mean"] for n in LEVELS]
        else:
            encodec_vals = [encodec_reports[n]["best_checkpoint"]["aggregate_latent_metrics"][key]["mean"] for n in LEVELS]
            dac_vals = [dac_reports[n]["final_aggregate_latent_metrics"][key]["mean"] for n in LEVELS]
        ax.plot(LEVELS, dac_vals, "o--", color="tab:red", label="DAC")
        ax.plot(LEVELS, encodec_vals, "o-", color="tab:blue", label="EnCodec")
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(LEVELS)
        ax.set_xticklabels([str(n) for n in LEVELS])
        ax.set_xlabel("N (training examples)")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Figure 2: DAC vs. EnCodec target scaling (same architecture/loss/optimizer/compute)")
    fig.tight_layout()
    out = ENCODEC_DIR / "figure2_dac_vs_encodec_scaling.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def figure3_oracle_relative_audio_quality(encodec_reports: dict, dac_reports: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("waveform_correlation_to_oracle", "waveform_correlation", "Waveform Correlation", axes[0], False),
        ("stft_distance_to_oracle", "stft_distance", "STFT Distance", axes[1], True),
    ]
    width = 0.18
    x = range(len(LEVELS))
    for enc_key, oracle_key, title, ax, is_distance in metrics:
        dac_oracle = [encodec_reports[n]["dac_oracle_aggregate"][oracle_key]["mean"] for n in LEVELS]
        dac_translator = [dac_reports[n]["final_aggregate_fixed_subset_audio_metrics"][enc_key]["mean"] for n in LEVELS]
        enc_oracle = [encodec_reports[n]["oracle_aggregate"][oracle_key]["mean"] for n in LEVELS]
        enc_translator = [encodec_reports[n]["best_checkpoint"]["aggregate_audio_metrics"][enc_key]["mean"] for n in LEVELS]

        ax.bar([i - 1.5 * width for i in x], dac_oracle, width, label="DAC oracle", color="tab:red", alpha=0.5)
        ax.bar([i - 0.5 * width for i in x], dac_translator, width, label="DAC translator", color="tab:red")
        ax.bar([i + 0.5 * width for i in x], enc_oracle, width, label="EnCodec oracle", color="tab:blue", alpha=0.5)
        ax.bar([i + 1.5 * width for i in x], enc_translator, width, label="EnCodec translator", color="tab:blue")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"N={n}" for n in LEVELS])
        ax.set_title(title + (" (lower=better)" if is_distance else " (higher=better)"))
        ax.legend(fontsize=7)

    # log-mel: only EnCodec oracle/DAC oracle are directly available; translator log-mel not tracked
    # per-level in evaluate_prediction's summary (only waveform_correlation/stft), so this panel
    # shows the two codecs' own oracle ceilings only, clearly labeled.
    ax = axes[2]
    dac_oracle_mel = [encodec_reports[n]["dac_oracle_aggregate"]["logmel_cosine_similarity"]["mean"] for n in LEVELS]
    enc_oracle_mel = [encodec_reports[n]["oracle_aggregate"]["logmel_cosine_similarity"]["mean"] for n in LEVELS]
    ax.plot(LEVELS, dac_oracle_mel, "o--", color="tab:red", label="DAC oracle")
    ax.plot(LEVELS, enc_oracle_mel, "o--", color="tab:blue", label="EnCodec oracle")
    ax.set_xscale("log", base=2)
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([str(n) for n in LEVELS])
    ax.set_title("Log-Mel Cosine Similarity\n(codec oracle ceilings only -- translator log-mel not tracked per-level)")
    ax.legend(fontsize=7)

    fig.suptitle("Figure 3: oracle-relative audio quality -- DAC and EnCodec have different reconstruction ceilings")
    fig.tight_layout()
    out = ENCODEC_DIR / "figure3_oracle_relative_audio_quality.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def figure4_per_example_n32(encodec_reports: dict, dac_reports: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    enc_pe_latent = encodec_reports[32]["best_checkpoint"]["per_example_latent_metrics"]
    dac_pe_latent = dac_reports[32]["final_per_example_latent_metrics"]
    common_tracks = sorted(set(enc_pe_latent) & set(dac_pe_latent))
    enc_cosine = [enc_pe_latent[t]["cosine"] for t in common_tracks]
    dac_cosine = [dac_pe_latent[t]["cosine"] for t in common_tracks]
    order = sorted(range(len(common_tracks)), key=lambda i: dac_cosine[i])
    axes[0].plot(range(len(common_tracks)), [dac_cosine[i] for i in order], "o-", color="tab:red", label="DAC", markersize=4)
    axes[0].plot(range(len(common_tracks)), [enc_cosine[i] for i in order], "o-", color="tab:blue", label="EnCodec", markersize=4)
    axes[0].set_xticks(range(len(common_tracks)))
    axes[0].set_xticklabels([common_tracks[i] for i in order], rotation=90, fontsize=6)
    axes[0].set_ylabel("Latent cosine similarity")
    axes[0].set_title("Per-example latent cosine, N=32\n(sorted by DAC value)")
    axes[0].legend()

    enc_pe_audio = encodec_reports[32]["best_checkpoint"]["per_example_audio_metrics"]
    enc_wav_corr = [enc_pe_audio[t]["waveform_correlation_to_oracle"] for t in common_tracks]
    order2 = sorted(range(len(common_tracks)), key=lambda i: enc_wav_corr[i])
    colors = ["tab:red" if i == order2[0] else "tab:green" if i == order2[-1] else "tab:blue" for i in range(len(order2))]
    axes[1].bar(range(len(common_tracks)), [enc_wav_corr[i] for i in order2], color=colors)
    axes[1].set_xticks(range(len(common_tracks)))
    axes[1].set_xticklabels([common_tracks[i] for i in order2], rotation=90, fontsize=6)
    axes[1].set_ylabel("Waveform correlation (pred vs. target-decode)")
    axes[1].set_title("Per-example EnCodec waveform correlation, N=32\n(DAC per-example audio only available for the 2-track fixed subset -- not shown)")

    fig.suptitle("Figure 4: per-example N=32 results -- does EnCodec lift all examples or an easier subset?")
    fig.tight_layout()
    out = ENCODEC_DIR / "figure4_per_example_n32.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    encodec_reports, dac_reports = load_reports()
    figure1_encodec_scaling_curve(encodec_reports)
    figure2_dac_vs_encodec_scaling(encodec_reports, dac_reports)
    figure3_oracle_relative_audio_quality(encodec_reports, dac_reports)
    figure4_per_example_n32(encodec_reports, dac_reports)


if __name__ == "__main__":
    main()
