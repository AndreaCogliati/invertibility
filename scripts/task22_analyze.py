"""TASK-22: first-block (L1) vs last_hidden_state (L12 = TASK-21) at matched steps and within the same window.

    uv run python scripts/task22_analyze.py --t22 artifacts/task22/task22_L1_v1 \
        --t21eval artifacts/task22/task22_T21eval_v1 [--t21 artifacts/task21/task21_N512ext_v1] \
        [--window 200000] [--out artifacts/task22/analysis_v1] [--sacct-hours H1 H2]

Evaluates nothing new. L1 rows come from the TASK-22 job (best, final = 200k, milestones); L12 rows from
the TASK-21 baseline evaluation job (its best = step 60,000 and milestones). TASK-21's window-best step
is found in its own validation history (steps <= --window), not assumed; if no evaluated checkpoint
exists at that step, the comparison is reported as unavailable. Paired by crop, track-clustered 95%
CIs; positive = L1 better (lower mel L1). Single seed; reused evaluation sets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muq_inversion.task19 import paired_arm_delta
from muq_inversion.task22 import checkpoint_steps, label_at_step, window_best

REPO_ROOT = Path(__file__).resolve().parents[1]
SETS = [("val (primary validation, unseen tracks)", ("val",)), ("test (primary test, unseen tracks)", ("test",)),
        ("val2 (secondary validation, unseen tracks)", ("val2",)), ("test2 (secondary test, unseen tracks)", ("test2",)),
        ("unseen crops of training tracks (core 32)", ("train_unseen",)),
        ("unseen crops of training tracks (96)", ("train_unseen_new",)),
        ("memorized crops (core 32)", ("memorized",)), ("memorized crops (96)", ("memorized_new",))]
HELDOUT = ("val", "test", "val2", "test2")
MATCHED = (10000, 20000, 50000, 60000, 100000, 200000)
SAME = ("track_manifest_sha256", "crop_manifest_sha256", "bigvgan_generator_sha256", "num_pool_crops", "n_train_tracks",
        "translator_parameters", "seed", "batch_size", "learning_rate", "adam_betas", "eval_crops", "hidden_channels",
        "lr_schedule", "eval_schedule", "loss", "muq_model_id")
CONTROLS = ("shuffled_across", "shuffled_time", "constant_zero", "average_mel")


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t22", type=Path, required=True)
    parser.add_argument("--t21eval", type=Path, required=True)
    parser.add_argument("--t21", type=Path, default=Path("artifacts/task21/task21_N512ext_v1"))
    parser.add_argument("--window", type=int, default=200000)
    parser.add_argument("--matched", type=int, nargs="*", default=list(MATCHED))
    parser.add_argument("--out", type=Path, default=Path("artifacts/task22/analysis_v1"))
    parser.add_argument("--billing", type=float, default=8.0)
    parser.add_argument("--sacct-hours", type=float, nargs="*", default=None, help="elapsed hours of the TASK-22 jobs (sacct)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    out = (REPO_ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t22 = json.loads((REPO_ROOT / args.t22 / "report.json").read_text())
    t21e = json.loads((REPO_ROOT / args.t21eval / "report.json").read_text())
    t21 = json.loads((REPO_ROOT / args.t21 / "report.json").read_text())

    # 1. Same data, crops, model, recipe, schedule -- only the representation differs.
    rc22, rc21 = t22["resolved_config"], t21["resolved_config"]
    mismatch = {k: (rc21.get(k), rc22.get(k)) for k in SAME if rc21.get(k) != rc22.get(k)}
    if mismatch:
        raise SystemExit(f"STOP: TASK-22 and TASK-21 differ beyond the representation: {mismatch}")
    layer = rc22.get("muq_feature")
    if not layer or layer["hidden_state_index"] != 1 or rc21.get("muq_feature"):
        raise SystemExit("STOP: expected TASK-22 = hidden_states[1] and TASK-21 = last_hidden_state")
    notes = [] if rc21.get("uv_lock_sha256") == rc22.get("uv_lock_sha256") else ["uv.lock differs between TASK-21 and TASK-22"]

    rows = [{**r, "arm": f"L1@{r['checkpoint']}", "checkpoint": "best"} for r in read_csv(REPO_ROOT / args.t22 / "per_example.csv")]
    rows += [{**r, "arm": f"L12@{r['checkpoint']}", "checkpoint": "best"} for r in read_csv(REPO_ROOT / args.t21eval / "per_example.csv")]
    crops = {a: {(r["eval_set"], r["crop_id"]) for r in rows if r["arm"] == a and r["condition"] == "correct"} for a in {r["arm"] for r in rows}}
    if len({frozenset(v) for v in crops.values()}) != 1:
        raise SystemExit("STOP: evaluation crops differ between checkpoints")

    # 2. Reproducibility of the baseline evaluation vs TASK-21's own in-job evaluation of the same best.pt.
    orig = {(r["eval_set"], r["crop_id"], r["condition"]): float(r["mel_l1"]) for r in read_csv(REPO_ROOT / args.t21 / "per_example.csv")
            if r["checkpoint"] == "best"}
    rep = [abs(float(r["mel_l1"]) - orig[(r["eval_set"], r["crop_id"], r["condition"])]) for r in rows
           if r["arm"] == "L12@best" and (r["eval_set"], r["crop_id"], r["condition"]) in orig]
    repro = {"n": len(rep), "max_abs_diff_mel_l1": max(rep) if rep else None,
             "same_best_step": t21e["best_checkpoint"]["step"] == t21["best_checkpoint"]["step"]}

    # 3. Which checkpoints exist at which steps; window-best of each run from its own history.
    s22, s21 = checkpoint_steps(t22), checkpoint_steps(t21e)
    wb22, wb21 = window_best(t22["history"], args.window), window_best(t21["history"], args.window)
    lab22, lab21 = label_at_step(s22, wb22["step"]), label_at_step(s21, wb21["step"])
    pairs, unavailable = [], []
    for s in args.matched:
        a, b = label_at_step(s22, s), label_at_step(s21, s)
        if a and b:
            pairs.append((f"matched step {s:,}", a, b, s, s))
        else:
            unavailable.append({"step": s, "L1": a, "L12": b})
    if lab22 and lab21:
        pairs.append((f"window best (<= {args.window:,} steps)", lab22, lab21, wb22["step"], wb21["step"]))
    else:
        unavailable.append({"window_best": True, "L1_step": wb22["step"], "L1": lab22, "L12_step": wb21["step"], "L12": lab21})

    def mean(arm, sets, key="mel_l1", cond="correct"):
        v = [float(r[key]) for r in rows if r["arm"] == arm and r["eval_set"] in sets and r["condition"] == cond]
        return float(np.mean(v)) if v else float("nan")

    comparisons = []
    for name, a, b, sa, sb in pairs:
        arm_l1, arm_l12 = f"L1@{a}", f"L12@{b}"
        per_set = {}
        for label, sets in SETS:
            d = paired_arm_delta(rows, arm_l12, arm_l1, sets, num_bootstrap=args.bootstrap)
            per_set[label] = {**d, "round_trip": mean(arm_l1, sets, "oracle_round_trip_mel_l1")}
        tracks = {}
        for s in HELDOUT:
            by = {}
            for r in rows:
                if r["eval_set"] == s and r["condition"] == "correct" and r["arm"] in (arm_l1, arm_l12):
                    by.setdefault(r["track_id"], {}).setdefault(r["arm"], []).append(float(r["mel_l1"]))
            tracks[s] = sorted(({"track_id": t, "genre": next(r["genre"] for r in rows if r["track_id"] == t),
                                 "L1": float(np.mean(v[arm_l1])), "L12": float(np.mean(v[arm_l12])),
                                 "L12_minus_L1": float(np.mean(v[arm_l12]) - np.mean(v[arm_l1]))} for t, v in by.items()),
                               key=lambda e: -e["L12_minus_L1"])
        comparisons.append({"name": name, "L1": {"label": a, "step": sa}, "L12": {"label": b, "step": sb},
                            "per_set": per_set, "per_track": tracks})

    def controls(report, label):
        agg = report["best"] if label == "best" else report["final"] if label == "final" else report["extra_checkpoints"][label]["aggregates"]
        return {s: {c: agg[s]["paired_mel_l1"][c] for c in CONTROLS} for s in HELDOUT + ("train_unseen", "memorized")}

    ctrl = {"L1": controls(t22, lab22) if lab22 else None, "L12": controls(t21e, lab21) if lab21 else None}

    # 4. Curves within the window (same validation schedule for both).
    h22 = [e for e in t22["history"] if e["step"] <= args.window]
    h21 = [e for e in t21["history"] if e["step"] <= args.window]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    keys = (("memorized/correct/mel_l1", "memorized crops (core)"), ("train_unseen/correct/mel_l1", "unseen crops of training tracks (core)"),
            ("val/correct/mel_l1", "primary val (selection)"))
    colors = {"L1": "tab:red", "L12": "tab:blue"}
    for ax, (k, title) in zip(axes, keys, strict=True):
        for tag, h in (("L1", h22), ("L12", h21)):
            ax.plot([e["step"] for e in h], [e[k] for e in h], color=colors[tag], lw=0.9,
                    label="first block (TASK-22)" if tag == "L1" else "last_hidden_state (TASK-21)")
        ax.axvline(wb22["step"], color=colors["L1"], ls=":", lw=1)
        ax.axvline(wb21["step"], color=colors["L12"], ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("optimizer step (log)")
    axes[0].set_ylabel("mel L1 (correct conditioning)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)
    v21 = {e["step"]: e["val/correct/mel_l1"] for e in h21}
    joint = [(e["step"], v21[e["step"]] - e["val/correct/mel_l1"]) for e in h22 if e["step"] in v21]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot([s for s, _ in joint], [d for _, d in joint], lw=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("optimizer step (log)")
    ax.set_ylabel("val mel L1: L12 - L1 (positive = L1 better)")
    fig.tight_layout()
    fig.savefig(out / "val_difference.png", dpi=110)
    plt.close(fig)
    diffs = np.array([d for _, d in joint])

    last = t22["history"][-1]
    run = {"best_step": t22["best_checkpoint"]["step"], "final_step": t22["final_step"],
           "window_best": {"L1": {"step": wb22["step"], "val": wb22["val/correct/mel_l1"]},
                           "L12": {"step": wb21["step"], "val": wb21["val/correct/mel_l1"]}},
           "exposure": {"draws_per_track": last.get("exposure/track_exposure_mean"), "pool_passes": last.get("diag/pool_passes"),
                        "track_cycles": last.get("diag/track_cycles"), "runtime_hours": last.get("diag/runtime_hours_cumulative"),
                        "steps_per_second_last": last.get("diag/steps_per_second")},
           "layer": {k: layer[k] for k in ("name", "hidden_state_index", "module", "feature_shape_per_crop", "frames_per_second")},
           "index_check": layer.get("index_check"),
           "val_difference_over_matched_checks": {"n": int(diffs.size), "mean": float(diffs.mean()) if diffs.size else None,
                                                  "fraction_L1_better": float((diffs > 0).mean()) if diffs.size else None},
           "allocation": None if args.sacct_hours is None else {"sacct_hours": args.sacct_hours,
                                                                "billing_hours": sum(args.sacct_hours) * args.billing}}
    analysis = {"run": run, "comparisons": comparisons, "unavailable": unavailable, "controls_window_best": ctrl,
                "baseline_reproducibility": repro, "notes": notes, "extra_unavailable": {
                    "L1": [k for k, e in (t22.get("extra_checkpoints") or {}).items() if not e["available"]],
                    "L12": [k for k, e in (t21e.get("extra_checkpoints") or {}).items() if not e["available"]]}}
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (out / "TASK-22_ANALYSIS.md").write_text(render(analysis))
    print(render(analysis))
    return 0


def render(a: dict) -> str:
    r = a["run"]
    lines = ["# TASK-22 analysis: first conformer block (L1) vs last_hidden_state (L12, TASK-21)", "",
             (f"- Representation: `{r['layer']['name']}` = `hidden_states[{r['layer']['hidden_state_index']}]` "
              f"(`{r['layer']['module']}`), {r['layer']['feature_shape_per_crop']} per crop, {r['layer']['frames_per_second']:.2f} frames/s."),
             (f"- L1 best step {r['best_step']:,} (final {r['final_step']:,}). Window best (own history): "
              f"L1 step {r['window_best']['L1']['step']:,} (val {_f(r['window_best']['L1']['val'], 4)}), "
              f"L12 step {r['window_best']['L12']['step']:,} (val {_f(r['window_best']['L12']['val'], 4)})."),
             f"- Exposure (L1): {r['exposure']}.",
             (f"- Primary val over all matched validation checks (L12 - L1): mean {_f(r['val_difference_over_matched_checks']['mean'], 4)}, "
              f"L1 lower at {_f(100 * (r['val_difference_over_matched_checks']['fraction_L1_better'] or float('nan')), 0)}% of "
              f"{r['val_difference_over_matched_checks']['n']} checks (curve-level, unpaired-by-crop, descriptive)."),
             f"- Baseline evaluation reproduces TASK-21's own evaluation of best.pt: {a['baseline_reproducibility']}.",
             f"- Unavailable comparisons: {a['unavailable'] or 'none'}; unavailable milestones: {a['extra_unavailable']}.",
             f"- Allocation: {r['allocation']}. Notes: {a['notes'] or 'none'}.", "",
             ("Single seed; paired track-clustered CIs do not measure seed variability. val/test/val2/test2 are reused "
              "evaluation sets, not untouched confirmation sets; only primary val was used for selection. An L1 advantage "
              "shows greater recoverability with this translator and recipe, not that deeper layers discarded information; "
              "a null result does not establish equal information content."), ""]
    for c in a["comparisons"]:
        lines += [f"## {c['name']}: L1 `{c['L1']['label']}` (step {c['L1']['step']:,}) vs L12 `{c['L12']['label']}` (step {c['L12']['step']:,})", "",
                  "| set | round trip | L12 | L1 | relative improvement (L1 vs L12) | Δ [95% CI] | tracks L1 better |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for label, d in c["per_set"].items():
            lines.append(f"| {label} | {_f(d['round_trip'])} | {_f(d['mean_a'])} | {_f(d['mean_b'])} | {100 * d['relative_improvement']:.1f}% | "
                         f"{_f(d['mean_delta'])} [{_f(d['ci_low'])}, {_f(d['ci_high'])}] | "
                         f"{100 * d['proportion_tracks_positive']:.0f}% of {d['n_tracks']} |")
        lines += ["", "Per-track held-out differences (mean over the track's crops; L12 - L1, positive = L1 better):", ""]
        for s, ts in c["per_track"].items():
            lines.append(f"- {s}: " + ", ".join(f"{t['track_id']} {t['L12_minus_L1']:+.3f}" for t in ts))
        lines.append("")
    lines += ["## Controls at each run's window-best checkpoint (control - correct mel L1, [95% CI])", "",
              "| run | set | " + " | ".join(CONTROLS) + " |", "| --- | --- | " + " | ".join("---" for _ in CONTROLS) + " |"]
    for run, cs in a["controls_window_best"].items():
        for s, d in (cs or {}).items():
            lines.append(f"| {run} | {s} | " + " | ".join(f"{_f(d[c]['mean_delta'])} [{_f(d[c]['ci_low'])}, {_f(d[c]['ci_high'])}]"
                                                        for c in CONTROLS) + " |")
    lines += ["", ("Curves: `curves.png` (memorized, unseen crops of training tracks, primary val; dotted = window best), "
                   "`val_difference.png` (L12 - L1 primary val at every shared validation step)."), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
