"""TASK-05's two required deliverable figures, built from `scripts/train_gate3_scale_wandb.py`'s
`artifacts/task05_gate3_scale/scaling_summary.json` (the five levels' best-by-MSE-checkpoint
aggregate metrics).

1. **Main deliverable**: training MSE, cosine similarity, norm ratio, variance ratio, and
   waveform correlation vs. number of training examples (log-x axis, since N spans 2-512).
2. **Additional required figure**: N vs. norm ratio and N vs. variance ratio, each with a
   horizontal reference line at 1.0 (no collapse) to make the collapse visually unambiguous.

Read-only over the existing report; does not retrain or touch any run/checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "artifacts/task05_gate3_scale/scaling_summary.json"
FIGURE_DIR = REPO_ROOT / "artifacts/task05_gate3_scale"


def main() -> None:
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    rows = sorted(summary["rows"], key=lambda r: r["n"])
    n_values = [r["n"] for r in rows]

    # --- Main deliverable: 5-panel scaling figure ---
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    panels = [
        ("mse_mean", "Training MSE", axes[0], True),
        ("cosine_mean", "Cosine Similarity", axes[1], False),
        ("norm_ratio_mean", "Norm Ratio", axes[2], False),
        ("variance_ratio_mean", "Variance Ratio", axes[3], False),
        ("waveform_correlation_to_oracle_mean", "Waveform Correlation", axes[4], False),
    ]
    for key, title, ax, logy in panels:
        values = [r[key] for r in rows]
        ax.plot(n_values, values, marker="o", color="#2b6cb0", linewidth=2)
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Number of training examples (N)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values])
    fig.suptitle("TASK-05: Spectrogram-to-DAC Gate-3 scaling curve (best-by-MSE checkpoint per level)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    main_path = FIGURE_DIR / "scaling_curve_main.png"
    fig.savefig(main_path, dpi=150)
    plt.close(fig)
    print(f"wrote {main_path}")

    # --- Additional required figure: norm ratio and variance ratio vs N, with a 1.0 reference line ---
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in [(ax1, "norm_ratio_mean", "Number of Training Examples vs. Norm Ratio"),
                            (ax2, "variance_ratio_mean", "Number of Training Examples vs. Variance Ratio")]:
        values = [r[key] for r in rows]
        ax.plot(n_values, values, marker="o", color="#c05621", linewidth=2, markersize=8)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="1.0 = no collapse")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of training examples (N)")
        ax.set_ylabel(title.split(" vs. ")[1])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values])
        ax.legend()
    fig2.tight_layout()
    additional_path = FIGURE_DIR / "scaling_curve_norm_variance_ratio.png"
    fig2.savefig(additional_path, dpi=150)
    plt.close(fig2)
    print(f"wrote {additional_path}")


if __name__ == "__main__":
    main()
