"""TASK-21: analyze the extended N512 continuation against TASK-20 best, A128, and A32.

    uv run python scripts/task21_analyze.py --t21 artifacts/task21/task21_N512ext_v1_eval \
        [--t20 artifacts/task20/task20_N512_v1] [--a128 ...] [--a32 ...] [--out artifacts/task21/analysis_v1] \
        [--billing 8] [--sacct-hours H]

Evaluates nothing new: reads the evaluation job's report/per_example and the reference arms' files.
Every set is reported separately (val, test, val2, test2; unseen crops of training tracks; memorized).
Paired by crop with track-clustered 95% CIs. These do not measure seed variability. Test sets are
reused evaluation sets (seen in TASK-18..20 reports), not untouched confirmation sets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np

from muq_inversion.task19 import paired_arm_delta

REPO_ROOT = Path(__file__).resolve().parents[1]
SETS = [("val", ("val",)), ("test", ("test",)), ("val2", ("val2",)), ("test2", ("test2",)),
        ("unseen crops of training tracks (core 32)", ("train_unseen",)),
        ("unseen crops of training tracks (96)", ("train_unseen_new",)),
        ("memorized (core 32)", ("memorized",)), ("memorized (96)", ("memorized_new",))]
COMPARISONS = [("T20best", "T21best"), ("T20best", "T21final"), ("A128", "T21best"), ("A32", "T21best")]
SPIKE_JUMP = 0.03  # a validation check at least this much above the previous one (mel L1)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t21", type=Path, required=True)
    parser.add_argument("--t20", type=Path, default=Path("artifacts/task20/task20_N512_v1"))
    parser.add_argument("--a128", type=Path, default=Path("artifacts/task19/task19_A128_v1"))
    parser.add_argument("--a32", type=Path, default=Path("artifacts/task19/task19_A32_v1"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/task21/analysis_v1"))
    parser.add_argument("--billing", type=float, default=8.0, help="sacct billing units per job-hour (verified: 8)")
    parser.add_argument("--sacct-hours", type=float, default=None, help="total elapsed hours of all TASK-21 jobs (sacct)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    out = (REPO_ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t21 = json.loads((REPO_ROOT / args.t21 / "report.json").read_text())

    rows = []
    for r in read_csv(REPO_ROOT / args.t21 / "per_example.csv"):
        rows.append({**r, "arm": "T21best" if r["checkpoint"] == "best" else "T21final", "checkpoint": "best"})
    for arm, d in (("T20best", args.t20), ("A128", args.a128), ("A32", args.a32)):
        rows += [{**r, "arm": arm} for r in read_csv(REPO_ROOT / d / "per_example.csv") if r["checkpoint"] == "best"]
    crops = {a: {(r["eval_set"], r["crop_id"]) for r in rows if r["arm"] == a and r["condition"] == "correct"}
             for a in ("T21best", "T21final", "T20best", "A128", "A32")}
    if len({frozenset(v) for v in crops.values()}) != 1:
        raise SystemExit("STOP: evaluation crops differ between arms")

    def mean(arm, sets, key="mel_l1", cond="correct"):
        v = [float(r[key]) for r in rows if r["arm"] == arm and r["eval_set"] in sets and r["condition"] == cond]
        return float(np.mean(v)) if v else float("nan")

    table = {label: {"round_trip": mean("T21best", sets, "oracle_round_trip_mel_l1"),
                     **{a: mean(a, sets) for a in ("A32", "A128", "T20best", "T21best", "T21final")}}
             for label, sets in SETS}
    paired = {label: {f"{b}_vs_{a}": paired_arm_delta(rows, a, b, sets, num_bootstrap=args.bootstrap) for a, b in COMPARISONS}
              for label, sets in SETS}

    hist = t21["history"]
    cont = t21.get("continuation") or {}
    parent_step = ((cont.get("parent") or {}).get("latest") or {}).get("step", 4250)
    vals = [(e["step"], e["val/correct/mel_l1"]) for e in hist]
    spikes = []
    for (s0, v0), (s1, v1) in itertools.pairwise(vals):
        if v1 - v0 >= SPIKE_JUMP:
            e = next(x for x in hist if x["step"] == s1)
            near = [ev for ev in (cont.get("events") or []) if s0 <= ev["step"] <= s1]
            spikes.append({"step": s1, "val_before": v0, "val": v1, "train_l1": e.get("train_l1"),
                           "train_l1_window_max": e.get("diag/train_l1_window_max"),
                           "grad_norm_window_max": e.get("diag/grad_norm_window_max"),
                           "track_cycles": e.get("diag/track_cycles"), "events_in_interval": near})
    resumes = [ev["step"] for ev in (cont.get("events") or []) if ev["event"] in ("resume", "continuation_start")]
    best_step, final_step = t21["best_checkpoint"]["step"], t21["final_step"]
    last = hist[-1]
    runtime_h = (cont.get("runtime_seconds") or 0) / 3600
    passes = {"optimizer_steps_in_continuation": final_step - parent_step, "total_optimizer_steps": final_step,
              "draws_per_track": last.get("exposure/track_exposure_mean"),
              "track_cycles": last.get("diag/track_cycles"), "pool_passes": last.get("diag/pool_passes")}
    charge = {"billing_per_hour": args.billing, "runtime_budget_hours_counted": runtime_h,
              "sacct_hours": args.sacct_hours,
              "billing_hours": None if args.sacct_hours is None else args.sacct_hours * args.billing}

    # Plots: full continuation (log step axis) and the early region (<= 20,000 steps).
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
    for ax, (lo, hi, logx, title) in zip(axes, ((1, None, True, "full run (log step)"), (0, 20000, False, "early region (0-20k)")), strict=True):
        h = [e for e in hist if e["step"] >= lo and (hi is None or e["step"] <= hi)]
        for k, lab in (("memorized/correct/mel_l1", "memorized (core)"), ("train_unseen/correct/mel_l1", "unseen crops (core)"),
                       ("val/correct/mel_l1", "val (selection)")):
            ax.plot([e["step"] for e in h], [e[k] for e in h], label=lab)
        ax.axvline(2250, color="k", ls=":", lw=1, label="TASK-20 best (2,250)")
        ax.axvline(parent_step, color="grey", ls="--", lw=1, label=f"continuation start ({parent_step})")
        if best_step != 2250:
            ax.axvline(best_step, color="g", ls=":", lw=1.5, label=f"new best ({best_step})")
        for sp in spikes:
            if hi is None or sp["step"] <= hi:
                ax.plot(sp["step"], sp["val"], "rv")
        for rs in resumes:
            if (hi is None or rs <= hi) and rs > parent_step:
                ax.axvline(rs, color="orange", lw=0.8)
        if logx:
            ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
    axes[0].set_ylabel("mel L1 (correct conditioning)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)

    analysis = {"best_step": best_step, "final_step": final_step, "parent_step": parent_step,
                "best_is_task20_incumbent": best_step == 2250, "runtime": runtime_h, "exposure": passes,
                "allocation": charge, "spikes": spikes, "resumes": resumes, "table": table, "paired": paired}
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (out / "TASK-21_ANALYSIS.md").write_text(render(analysis))
    print(render(analysis))
    return 0


def render(a: dict) -> str:
    imp = a["paired"]["val"]["T21best_vs_T20best"]
    lines = [
        "# TASK-21 analysis: extended N512 continuation (no early stopping)", "",
        (f"- Best step **{a['best_step']}** ({'TASK-20 incumbent; no new best' if a['best_is_task20_incumbent'] else 'new best'}); "
        f"final step {a['final_step']}; continued from step {a['parent_step']}."),
        f"- Runtime counted toward the budget: {a['runtime']:.2f} h. Allocation: {a['allocation']}.",
        f"- Exposure: {a['exposure']}.",
        f"- Validation spikes (>= {SPIKE_JUMP} jump between checks): {len(a['spikes'])}; resumes at steps {a['resumes']}.",
        (f"- Primary val, best vs TASK-20 best: {100 * imp['relative_improvement']:.1f}% "
        f"[{_f(imp['ci_low'])}, {_f(imp['ci_high'])}]."), "",
        ("Paired track-clustered CIs do not measure seed variability. Test sets are reused evaluation sets, "
        "not untouched confirmation sets. A null result means only that extended training with this unchanged "
        "recipe did not improve the selected result."), "",
        "## Mean mel L1 (correct conditioning), each set separately", "",
        "| set | round trip | A32 | A128 | TASK-20 best | TASK-21 best | TASK-21 final |", "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, t in a["table"].items():
        lines.append(f"| {label} | {_f(t['round_trip'])} | {_f(t['A32'])} | {_f(t['A128'])} | {_f(t['T20best'])} | "
                     f"{_f(t['T21best'])} | {_f(t['T21final'])} |")
    lines += ["", "## Paired comparisons (positive = second-named arm better)", "",
              "| set | comparison | relative improvement | Δ [95% CI] | tracks improved |", "| --- | --- | --- | --- | --- |"]
    for label, ds in a["paired"].items():
        for k, d in ds.items():
            lines.append(f"| {label} | {k} | {100 * d['relative_improvement']:.1f}% | {_f(d['mean_delta'])} "
                         f"[{_f(d['ci_low'])}, {_f(d['ci_high'])}] | {100 * d['proportion_tracks_positive']:.0f}% of {d['n_tracks']} |")
    lines += ["", "## Validation spikes and coincident events", ""]
    lines += [f"- step {s['step']}: val {_f(s['val_before'])} -> {_f(s['val'])}; train L1 window max {_f(s['train_l1_window_max'])}, "
              f"grad-norm window max {_f(s['grad_norm_window_max'])}, track cycles {_f(s['track_cycles'], 1)}, "
              f"events {[e['event'] for e in s['events_in_interval']]}" for s in a["spikes"]] or ["- none"]
    lines += ["", ("Curves: `curves.png` (full run on a log axis; early region; dotted = TASK-20 best, dashed = continuation "
              "start, green = new best, red triangles = spikes, orange = resumes)."), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
