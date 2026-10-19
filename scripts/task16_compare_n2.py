"""TASK-16: compare the corrected N=2 run with the TASK-15 N=2 pipeline, best checkpoints.

Baseline: artifacts/task16_bigvgan_corrected/task15_baseline_rerun/report.json -- the TASK-15
config rerun after the TASK-16 code changes (bit-identical training trajectory and metrics to the
original TASK-15 report), which adds the diagnostics the original report lacks.

    uv run python scripts/task16_compare_n2.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "artifacts/task16_bigvgan_corrected/task15_baseline_rerun/report.json"
NEW = REPO / "artifacts/task16_bigvgan_corrected/n2/report.json"
OUT = REPO / "artifacts/task16_bigvgan_corrected/comparison_n2.json"

ROWS = [
    ("mel L1, raw pred vs target", lambda r: r["mel_l1_pred_vs_target"]),
    ("mel variance ratio, raw pred / target", lambda r: r["mel_variance_ratio"]),
    ("raw pred bins below mel floor", lambda r: r["raw_bins_below_mel_floor"]),
    ("bins clamped at decode", lambda r: r["bins_clamped_at_decode"]),
    ("pred-vs-oracle waveform corr", lambda r: r["pred_vs_oracle"]["waveform_correlation"]),
    ("pred-vs-oracle resynthesis mel L1", lambda r: r["pred_vs_oracle"]["resynthesis_mel_l1"]),
    ("pred-vs-oracle STFT distance", lambda r: r["pred_vs_oracle"]["stft_distance"]),
    ("resynthesis mel L1 vs original (oracle in brackets)",
     lambda r: f"{r['pred_vs_original']['resynthesis_mel_l1']:.4f} ({r['oracle_vs_original']['resynthesis_mel_l1']:.4f})"),
    ("STFT distance vs original (oracle in brackets)",
     lambda r: f"{r['pred_vs_original']['stft_distance']:.4f} ({r['oracle_vs_original']['stft_distance']:.4f})"),
    ("shuffled-across mel L1 vs target", lambda r: r["mel_l1_shuffled_vs_target"]),
    ("shuffled-across STFT vs original", lambda r: r["shuffled_vs_original"]["stft_distance"]),
    ("shuffled bins clamped at decode", lambda r: r["shuffled_bins_clamped_at_decode"]),
    ("all four criteria pass", lambda r: all(r["criteria"].values())),
]


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main() -> None:
    base, new = json.loads(BASE.read_text()), json.loads(NEW.read_text())
    crops = list(new["best"]["per_crop"])
    total_bins = 100 * new["geometry"]["mel_frames"]
    out = {"best_step": {"task15": base["best_checkpoint_step"], "task16": new["best_checkpoint_step"]},
           "bins_per_crop": total_bins, "per_crop": {}}
    lines = ["| metric | crop | TASK-15 pipeline | TASK-16 corrected |", "| --- | --- | --- | --- |"]
    for name, f in ROWS:
        for c in crops:
            b, n = f(base["best"]["per_crop"][c]), f(new["best"]["per_crop"][c])
            out["per_crop"].setdefault(c, {})[name] = {"task15": b, "task16": n}
            lines.append(f"| {name} | {c} | {fmt(b)} | {fmt(n)} |")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"best checkpoint step: TASK-15 {base['best_checkpoint_step']}, TASK-16 {new['best_checkpoint_step']}; "
          f"{total_bins} mel bins per crop")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
