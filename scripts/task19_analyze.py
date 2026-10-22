"""TASK-19: compare the four arms and apply the pre-registered interpretation table.

    uv run python scripts/task19_analyze.py \
        --arm A32=artifacts/task19/task19_A32_v1 --arm A128=artifacts/task19/task19_A128_v1 \
        --arm B32=artifacts/task19/task19_B32_v1 --arm B128=artifacts/task19/task19_B128_v1 \
        [--task18 artifacts/task18_heldout/task18_heldout_v1] [--out artifacts/task19/analysis_v1]

Reads each arm's report.json and per_example.csv. Writes analysis.json, per_example_all_arms.csv,
curves.png, and TASK-19_ANALYSIS.md. Paired comparisons use the same crops and a track-clustered
bootstrap; "much better" and the outcome codes are those in configs/task19.yaml.
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
import yaml

from muq_inversion.task19 import (
    ARMS,
    classify_arms,
    judge,
    load_faults,
    paired_arm_delta,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "primary_heldout": ("val", "test"),
    "secondary_heldout": ("val2", "test2"),
    "train_unseen_core": ("train_unseen",),
    "memorized_core": ("memorized",),
    "new_tracks_unseen_crops": ("train_unseen_new",),
}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_of(rows, arm, sets, checkpoint="best", condition="correct", key="mel_l1"):
    v = [float(r[key]) for r in rows if r["arm"] == arm and r["checkpoint"] == checkpoint
         and r["condition"] == condition and r["eval_set"] in sets and r[key] not in ("", "nan")]
    return float(np.mean(v)) if v else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, help="ARM=artifact_dir, once per arm")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/task19.yaml")
    parser.add_argument("--task18", type=Path, default=REPO_ROOT / "artifacts/task18_heldout/task18_heldout_v1")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/task19/analysis_v1")
    args = parser.parse_args()
    args.out = (REPO_ROOT / args.out).resolve()
    cfg = yaml.safe_load(args.config.read_text())
    dirs = dict(a.split("=", 1) for a in args.arm)
    if set(dirs) != set(ARMS):
        raise SystemExit(f"need exactly the arms {ARMS}, got {sorted(dirs)}")
    args.out.mkdir(parents=True, exist_ok=True)

    reports, rows = {}, []
    for arm in ARMS:
        d = REPO_ROOT / dirs[arm]
        reports[arm] = json.loads((d / "report.json").read_text())
        if reports[arm].get("arm") not in (arm, None):
            raise SystemExit(f"{d} is arm {reports[arm].get('arm')}, not {arm}")
        rows += [{**r, "arm": arm} for r in read_csv(d / "per_example.csv")]
    with open(args.out / "per_example_all_arms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)

    rule = cfg["arm_comparison"]
    nb = cfg["bootstrap_samples"]
    table, deltas = {}, {}
    for g, sets in GROUPS.items():
        table[g] = {arm: {"best": mean_of(rows, arm, sets), "final": mean_of(rows, arm, sets, "final"),
                          **{c: mean_of(rows, arm, sets, condition=c) for c in
                             ("shuffled_across", "shuffled_time", "constant_zero", "average_mel")},
                          "round_trip": mean_of(rows, arm, sets, key="oracle_round_trip_mel_l1"),
                          "best_step": reports[arm]["best_checkpoint"]["step"]} for arm in ARMS}
        deltas[g] = {f"{b}_vs_{a}": paired_arm_delta(rows, a, b, sets, num_bootstrap=nb)
                     for a, b in (("A32", "A128"), ("A32", "B32"), ("A32", "B128"), ("A128", "B128"), ("B32", "B128"))}
        for k, d in deltas[g].items():
            d["judgement"] = judge(d, rule)
    primary = {b: deltas["primary_heldout"][f"{b}_vs_A32"] for b in ("A128", "B32", "B128")}
    training = {b: deltas["memorized_core"][f"{b}_vs_A32"] for b in ("A128", "B32", "B128")}
    outcome = classify_arms(primary, training, rule)
    secondary_outcome = classify_arms({b: deltas["secondary_heldout"][f"{b}_vs_A32"] for b in ("A128", "B32", "B128")},
                                      training, rule)

    # Sanity: A32 re-evaluation vs TASK-18 (same checkpoint, same crops).
    sanity = {}
    t18 = args.task18 / "per_example.csv"
    if t18.exists():
        # Correct conditioning depends only on the checkpoint and the crop; controls also depend on
        # the evaluated set (shuffled-across partner, training-pool average), identical only when the
        # sets are (full runs).
        ref = {(r["crop_id"], r["condition"]): float(r["mel_l1"]) for r in read_csv(t18) if r["checkpoint"] == "best"}
        for cond_set, label in ((("correct",), "correct"), (None, "all_conditions")):
            diffs = [abs(float(r["mel_l1"]) - ref[(r["crop_id"], r["condition"])]) for r in rows
                     if r["arm"] == "A32" and r["checkpoint"] == "best" and (r["crop_id"], r["condition"]) in ref
                     and (cond_set is None or r["condition"] in cond_set)]
            sanity[label] = {"crops_compared": len(diffs), "max_abs_mel_l1_diff": float(max(diffs)) if diffs else None}

    # Secondary: held-out results by whether the track's artist appears in the arm's training tracks.
    faults = load_faults(REPO_ROOT / cfg["fault_list"])
    tracks = read_csv(REPO_ROOT / cfg["track_manifest"])
    train_ids = {arm: {t["track_id"] for t in tracks if t["split"] == "train" and (t["core"] == "1" or arm.endswith("128"))}
                 for arm in ARMS}
    artist_strata = {}
    for arm in ARMS:
        seen_artists = {faults["artist"].get(t) for t in train_ids[arm]} - {None}
        out = {}
        for label, sets in (("primary", ("val", "test")), ("secondary", ("val2", "test2"))):
            for stratum in ("artist_seen", "artist_not_seen_or_unknown"):
                v = [float(r["mel_l1"]) for r in rows if r["arm"] == arm and r["checkpoint"] == "best"
                     and r["condition"] == "correct" and r["eval_set"] in sets
                     and ((faults["artist"].get(r["track_id"]) in seen_artists) == (stratum == "artist_seen"))]
                out[f"{label}/{stratum}"] = {"mean_mel_l1": float(np.mean(v)) if v else float("nan"), "n_crops": len(v)}
        artist_strata[arm] = out

    # Curves.
    keys = [("memorized/correct/mel_l1", "memorized (core)"), ("train_unseen/correct/mel_l1", "unseen crops (core)"),
            ("val/correct/mel_l1", "val (primary)"), ("val2/correct/mel_l1", "val2 (secondary)")]
    fig, axes = plt.subplots(1, len(keys), figsize=(18, 4), sharey=True)
    for ax, (k, title) in zip(axes, keys, strict=True):
        for arm in ARMS:
            h = [e for e in reports[arm]["history"] if k in e]
            if h:  # fixed color per arm: A32 has no val2 curve, so the default cycle would shift
                ax.plot([e["step"] for e in h], [e[k] for e in h], label=arm, color=f"C{ARMS.index(arm)}")
        ax.set_title(title)
        ax.set_xlabel("step")
    axes[0].set_ylabel("mel L1 (correct conditioning)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.out / "curves.png", dpi=110)
    plt.close(fig)

    analysis = {"outcome_primary": outcome, "outcome_secondary_heldout": secondary_outcome, "table": table,
                "paired": deltas, "sanity": sanity, "artist_strata": artist_strata,
                "per_arm_task18_rule": {arm: reports[arm]["outcome_best"]["code"] for arm in ARMS},
                "wandb_runs": {arm: reports[arm]["wandb_run_id"] for arm in ARMS}, "rule": rule}
    (args.out / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (args.out / "TASK-19_ANALYSIS.md").write_text(render(analysis))
    print(render(analysis))
    return 0


def _f(x, nd=3):
    return "n/a" if x is None or np.isnan(x) else f"{x:.{nd}f}"


def render(a: dict) -> str:
    o = a["outcome_primary"]
    lines = [
        "# TASK-19 analysis: data x recipe", "",
        (f"**Pre-registered outcome (primary held-out, val + test): `{o['code']}`.** Judgements vs A32: "
         + ", ".join(f"{k} {v}" for k, v in o["judgements"].items())
         + (f"; generalization-gap flag: {o['generalization_gap_arms']}" if o["generalization_gap_arms"] else "") + "."),
        "",
        (f"Secondary held-out (val2 + test2), same rule: `{a['outcome_secondary_heldout']['code']}` "
         f"({', '.join(f'{k} {v}' for k, v in a['outcome_secondary_heldout']['judgements'].items())})."),
        "",
        f"A32 re-evaluation vs TASK-18: {a['sanity']}. W&B runs: {a['wandb_runs']}.",
        "",
        "## Mean mel L1 (correct conditioning), best checkpoint [final]", "",
        "| group | round trip | " + " | ".join(ARMS) + " | best steps |",
        "| --- | --- | " + " | ".join("---" for _ in ARMS) + " | --- |",
    ]
    for g, t in a["table"].items():
        lines.append(f"| {g} | {_f(t['A32']['round_trip'])} | "
                     + " | ".join(f"{_f(t[arm]['best'])} [{_f(t[arm]['final'])}]" for arm in ARMS)
                     + " | " + ", ".join(f"{arm} {t[arm]['best_step']}" for arm in ARMS) + " |")
    lines += ["", "## Paired comparisons (positive = second arm better; track-clustered 95% CI)", "",
              "| group | comparison | relative improvement | Δ mel L1 [CI] | tracks improved | judgement |",
              "| --- | --- | --- | --- | --- | --- |"]
    for g, ds in a["paired"].items():
        for k, d in ds.items():
            lines.append(f"| {g} | {k} | {100 * d['relative_improvement']:.1f}% | {_f(d['mean_delta'])} "
                         f"[{_f(d['ci_low'])}, {_f(d['ci_high'])}] | {100 * d['proportion_tracks_positive']:.0f}% "
                         f"({d['n_tracks']}) | {d['judgement']} |")
    lines += ["", "## Controls (best checkpoint, primary held-out)", "", "| arm | correct | shuffled_across | shuffled_time | constant_zero | average_mel |",
              "| --- | --- | --- | --- | --- | --- |"]
    t = a["table"]["primary_heldout"]
    lines += [f"| {arm} | {_f(t[arm]['best'])} | {_f(t[arm]['shuffled_across'])} | {_f(t[arm]['shuffled_time'])} | "
              f"{_f(t[arm]['constant_zero'])} | {_f(t[arm]['average_mel'])} |" for arm in ARMS]
    lines += ["", "## Secondary: held-out mel L1 by artist overlap with the arm's training tracks (Sturm 2013 artists)", "",
              "| arm | " + " | ".join(a["artist_strata"]["A32"]) + " |", "| --- | " + " | ".join("---" for _ in a["artist_strata"]["A32"]) + " |"]
    lines += [f"| {arm} | " + " | ".join(f"{_f(v['mean_mel_l1'])} (n={v['n_crops']})" for v in a["artist_strata"][arm].values()) + " |"
              for arm in ARMS]
    lines += ["", f"Per-arm TASK-18 rule codes: {a['per_arm_task18_rule']}.", "", "Curves: `curves.png`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
