"""tasks/TASK-12.md's required compute-response figures, built from TASK-11's completed N=32
report (20k baseline, reused unchanged) and this task's own 80k/320k reports.

Read-only over the existing reports; does not retrain or touch any run/checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK11_REPORT = REPO_ROOT / "artifacts/task11_gate3_encodec_scale/n32/report.json"
TASK12_DIR = REPO_ROOT / "artifacts/task12_encodec_n32_compute_ablation"
BUDGETS = [20000, 80000, 320000]


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_reports() -> dict[int, dict]:
    r20 = _load(TASK11_REPORT)
    r80 = _load(TASK12_DIR / "budget_80000/report.json")
    r320 = _load(TASK12_DIR / "budget_320000/report.json")
    return {20000: r20, 80000: r80, 320000: r320}


def best_agg(reports: dict[int, dict], budget: int) -> dict:
    r = reports[budget]
    return r["best_checkpoint"]["aggregate_audio_metrics"] if budget == 20000 else r["best_checkpoint"]["aggregate_metrics"]


def best_per_example(reports: dict[int, dict], budget: int) -> dict:
    r = reports[budget]
    return r["best_checkpoint"]["per_example_audio_metrics"] if budget == 20000 else r["best_checkpoint"]["per_example_metrics"]


def oracle_wav_corr(reports: dict[int, dict]) -> float:
    return reports[20000]["oracle_aggregate"]["waveform_correlation"]["mean"]


def figure_compute_response(reports: dict[int, dict]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    oracle = oracle_wav_corr(reports)

    wc = [best_agg(reports, b)["waveform_correlation_to_oracle"]["mean"] for b in BUDGETS]
    oracle_rel = [100 * v / oracle for v in wc]
    axes[0].plot(BUDGETS, wc, "o-", color="tab:blue")
    axes[0].axhline(oracle, color="black", linewidth=0.5, linestyle=":", label="EnCodec oracle")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Training steps")
    axes[0].set_title("Waveform correlation")
    axes[0].legend(fontsize=7)

    axes[1].plot(BUDGETS, oracle_rel, "o-", color="tab:green")
    axes[1].axhline(100, color="black", linewidth=0.5, linestyle=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Training steps")
    axes[1].set_ylabel("%")
    axes[1].set_title("Oracle-relative recovery")

    mse = [best_agg(reports, b)["mse"]["mean"] for b in BUDGETS]
    axes[2].plot(BUDGETS, mse, "o-", color="tab:red")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Training steps")
    axes[2].set_title("Latent MSE")

    var_ratio = [best_agg(reports, b)["variance_ratio"]["mean"] for b in BUDGETS]
    norm_ratio = [best_agg(reports, b)["norm_ratio"]["mean"] for b in BUDGETS]
    axes[3].plot(BUDGETS, var_ratio, "o-", color="tab:purple", label="variance ratio")
    axes[3].plot(BUDGETS, norm_ratio, "o-", color="tab:orange", label="norm ratio")
    axes[3].axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    axes[3].set_xscale("log")
    axes[3].set_xlabel("Training steps")
    axes[3].set_title("Norm / variance ratio")
    axes[3].legend(fontsize=7)

    fig.suptitle("TASK-12: EnCodec N=32 compute-response curve (best-by-MSE checkpoint per budget)")
    fig.tight_layout()
    out = TASK12_DIR / "figure1_compute_response_curve.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def figure_marginal_gains(reports: dict[int, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    wc = {b: best_agg(reports, b)["waveform_correlation_to_oracle"]["mean"] for b in BUDGETS}
    transitions = ["20k->80k", "80k->320k", "20k->320k"]
    gains = [wc[80000] - wc[20000], wc[320000] - wc[80000], wc[320000] - wc[20000]]
    ax.bar(transitions, gains, color=["tab:blue", "tab:green", "tab:gray"])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Waveform correlation gain")
    ax.set_title("Marginal gains in waveform correlation")
    fig.tight_layout()
    out = TASK12_DIR / "figure2_marginal_gains.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def figure_rank_stability(reports: dict[int, dict]) -> None:
    pe20 = best_per_example(reports, 20000)
    pe80 = best_per_example(reports, 80000)
    pe320 = best_per_example(reports, 320000)
    tracks = sorted(pe20)
    wc20 = np.array([pe20[t]["waveform_correlation_to_oracle"] for t in tracks])
    wc80 = np.array([pe80[t]["waveform_correlation_to_oracle"] for t in tracks])
    wc320 = np.array([pe320[t]["waveform_correlation_to_oracle"] for t in tracks])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [("20k", wc20, "80k", wc80, axes[0]), ("80k", wc80, "320k", wc320, axes[1]), ("20k", wc20, "320k", wc320, axes[2])]
    for label_a, a, label_b, b, ax in pairs:
        ax.scatter(a, b, s=15, alpha=0.6, color="tab:blue")
        lims = [0, 1]
        ax.plot(lims, lims, color="black", linewidth=0.5, linestyle=":")
        rho = spearmanr(a, b).statistic
        ax.set_xlabel(f"{label_a} waveform correlation")
        ax.set_ylabel(f"{label_b} waveform correlation")
        ax.set_title(f"{label_a} vs {label_b}\nSpearman rho={rho:.3f}")

    fig.suptitle("Figure: per-example rank stability across compute budgets")
    fig.tight_layout()
    out = TASK12_DIR / "figure3_rank_stability.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # gain vs starting point
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))
    gain_20_80 = wc80 - wc20
    gain_80_320 = wc320 - wc80
    axes2[0].scatter(wc20, gain_20_80, s=15, alpha=0.6, color="tab:orange")
    axes2[0].set_xlabel("20k waveform correlation")
    axes2[0].set_ylabel("Gain (20k -> 80k)")
    r1 = np.corrcoef(wc20, gain_20_80)[0, 1]
    axes2[0].set_title(f"Does compute help hard examples more?\nPearson r={r1:.3f}")

    axes2[1].scatter(wc80, gain_80_320, s=15, alpha=0.6, color="tab:orange")
    axes2[1].set_xlabel("80k waveform correlation")
    axes2[1].set_ylabel("Gain (80k -> 320k)")
    r2 = np.corrcoef(wc80, gain_80_320)[0, 1]
    axes2[1].set_title(f"Does compute help hard examples more?\nPearson r={r2:.3f}")

    fig2.suptitle("Figure: per-example gain vs. starting performance")
    fig2.tight_layout()
    out2 = TASK12_DIR / "figure4_gain_vs_starting_point.png"
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")


def figure_latent_vs_audio(reports: dict[int, dict]) -> None:
    pe20 = best_per_example(reports, 20000)
    pe80 = best_per_example(reports, 80000)
    pe320 = best_per_example(reports, 320000)
    tracks = sorted(pe20)

    cosine_all, var_all, wc_all = [], [], []
    for pe in (pe20, pe80, pe320):
        for t in tracks:
            cosine_all.append(pe[t]["cosine"])
            var_all.append(pe[t]["variance_ratio"])
            wc_all.append(pe[t]["waveform_correlation_to_oracle"])
    cosine_all, var_all, wc_all = np.array(cosine_all), np.array(var_all), np.array(wc_all)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(cosine_all, wc_all, s=10, alpha=0.4, color="tab:blue")
    rho1 = spearmanr(cosine_all, wc_all).statistic
    axes[0].set_xlabel("Latent cosine similarity")
    axes[0].set_ylabel("Waveform correlation")
    axes[0].set_title(f"Latent cosine vs. audio\nSpearman rho={rho1:.3f} (all budgets pooled)")

    axes[1].scatter(var_all, wc_all, s=10, alpha=0.4, color="tab:purple")
    rho2 = spearmanr(var_all, wc_all).statistic
    axes[1].set_xlabel("Variance ratio")
    axes[1].set_ylabel("Waveform correlation")
    axes[1].set_title(f"Variance ratio vs. audio\nSpearman rho={rho2:.3f} (all budgets pooled)")

    fig.suptitle("Figure: does latent fit track audio reconstruction quality?")
    fig.tight_layout()
    out = TASK12_DIR / "figure5_latent_vs_audio.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    reports = load_reports()
    figure_compute_response(reports)
    figure_marginal_gains(reports)
    figure_rank_stability(reports)
    figure_latent_vs_audio(reports)


if __name__ == "__main__":
    main()
