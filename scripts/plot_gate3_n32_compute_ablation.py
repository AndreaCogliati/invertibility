"""tasks/TASK-06.md's required figure: per-example waveform correlation to oracle for all 32
examples in the N=32 manifest, one panel per compute budget (20k/80k/320k). The 20k budget is
TASK-05's reused n32 run, which only decoded audio for its 2-track fixed subset (not all 32) --
that panel is annotated accordingly rather than showing 30 fabricated points.

Read-only over the existing reports; does not retrain or touch any run/checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "artifacts/task06_n32_compute_ablation"

TASK05_N32_REPORT = REPO_ROOT / "artifacts/task05_gate3_scale/n32/report.json"
BUDGET_80K_REPORT = FIGURE_DIR / "budget_80000/report.json"
BUDGET_320K_REPORT = FIGURE_DIR / "budget_320000/report.json"


def main() -> None:
    with open(BUDGET_80K_REPORT) as f:
        r80k = json.load(f)
    with open(BUDGET_320K_REPORT) as f:
        r320k = json.load(f)
    with open(TASK05_N32_REPORT) as f:
        r20k = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    # --- 20k panel: only the 2-track fixed subset has decode-based waveform correlation ---
    ax = axes[0]
    fixed = r20k["final_aggregate_fixed_subset_audio_metrics"]["waveform_correlation_to_oracle"]
    ax.bar(["pop.00012", "disco.00049"], [fixed["max"], fixed["min"]], color="tab:gray")
    ax.set_title(f"20k (best step={r20k['best_checkpoint_step']})\nfixed 2-track subset only")
    ax.set_ylabel("Waveform correlation to oracle")
    ax.tick_params(axis="x", rotation=90)
    ax.axhline(0, color="black", linewidth=0.5)

    for ax, report, label in [(axes[1], r80k, "80k"), (axes[2], r320k, "320k")]:
        per_example = report["final_per_example_metrics"]
        items = sorted(per_example.items(), key=lambda kv: kv[1]["waveform_correlation_to_oracle"])
        tracks = [k for k, _ in items]
        values = [v["waveform_correlation_to_oracle"] for _, v in items]
        colors = ["tab:red" if i == 0 else "tab:green" if i == len(items) - 1 else "tab:blue" for i in range(len(items))]
        ax.bar(range(len(tracks)), values, color=colors)
        ax.set_xticks(range(len(tracks)))
        ax.set_xticklabels(tracks, rotation=90, fontsize=6)
        ax.set_title(f"{label} (best step={report['best_checkpoint_step']}, n=32)")
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("TASK-06: per-example waveform correlation to oracle, by compute budget (N=32 fixed)")
    fig.tight_layout()
    out_path = FIGURE_DIR / "per_example_waveform_correlation.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
