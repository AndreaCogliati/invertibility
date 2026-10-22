"""TASK-20: compare the N=512 run with TASK-19's A128 and A32 on identical crops.

    uv run python scripts/task20_analyze.py --n512 artifacts/task20/task20_N512_v1 \
        [--a128 artifacts/task19/task19_A128_v1] [--a32 artifacts/task19/task19_A32_v1] [--out artifacts/task20/analysis_v1]

Every eval set is reported separately (primary val and test, secondary val2 and test2; unseen crops
of training tracks; memorized crops). Combined val+test and val2+test2 appear only for comparison with
TASK-19. Paired by crop, track-clustered bootstrap 95% CI. Descriptive: there is no new outcome code.
Single seed: a scaling data point, not a measurement of MuQ's information ceiling.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A32", "A128", "N512")
# (label, eval sets, what the crops are for N512)
GROUPS = [
    ("val (primary)", ("val",), "held-out tracks; used for checkpoint selection and early stopping"),
    ("test (primary)", ("test",), "held-out tracks; never used for any decision"),
    ("val2 (secondary)", ("val2",), "held-out tracks; not used for any decision"),
    ("test2 (secondary)", ("test2",), "held-out tracks; never used for any decision"),
    ("train_unseen (core 32)", ("train_unseen",), "unseen crops of TRAINING tracks (all arms)"),
    ("train_unseen_new (96)", ("train_unseen_new",), "unseen crops of TRAINING tracks for A128/N512; unseen tracks for A32"),
    ("memorized (core 32)", ("memorized",), "training crops (all arms)"),
    ("memorized_new (96)", ("memorized_new",), "training crops for A128/N512; unseen tracks for A32"),
    ("primary val+test (TASK-19 comparison only)", ("val", "test"), "combined, for comparison with TASK-19"),
    ("secondary val2+test2 (TASK-19 comparison only)", ("val2", "test2"), "combined, for comparison with TASK-19"),
]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_of(rows, arm, sets, checkpoint="best", condition="correct", key="mel_l1"):
    v = [float(r[key]) for r in rows if r["arm"] == arm and r["checkpoint"] == checkpoint and r["condition"] == condition
         and r["eval_set"] in sets and r[key] not in ("", "nan")]
    return float(np.mean(v)) if v else float("nan")


def _f(x, nd=3):
    return "n/a" if x is None or np.isnan(x) else f"{x:.{nd}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n512", type=Path, required=True)
    parser.add_argument("--a128", type=Path, default=Path("artifacts/task19/task19_A128_v1"))
    parser.add_argument("--a32", type=Path, default=Path("artifacts/task19/task19_A32_v1"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/task20/analysis_v1"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    dirs = {"A32": REPO_ROOT / args.a32, "A128": REPO_ROOT / args.a128, "N512": REPO_ROOT / args.n512}
    out = (REPO_ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    reports = {a: json.loads((d / "report.json").read_text()) for a, d in dirs.items()}
    rows = [{**r, "arm": a} for a, d in dirs.items() for r in read_csv(d / "per_example.csv")]

    # Identical crops: the N512 crop set must equal each reference arm's on every eval set.
    crop_sets = {a: {(r["eval_set"], r["crop_id"]) for r in rows if r["arm"] == a and r["checkpoint"] == "best" and r["condition"] == "correct"}
                 for a in ARMS}
    if not (crop_sets["N512"] == crop_sets["A128"] == crop_sets["A32"]):
        raise SystemExit("STOP: eval crops differ between arms")

    table, paired = {}, {}
    for label, sets, meaning in GROUPS:
        table[label] = {"meaning": meaning, "n_crops": sum(1 for c in crop_sets["N512"] if c[0] in sets),
                        "round_trip": mean_of(rows, "N512", sets, key="oracle_round_trip_mel_l1"),
                        **{f"{a}_best": mean_of(rows, a, sets) for a in ARMS},
                        "N512_final": mean_of(rows, "N512", sets, "final"),
                        **{f"N512_{c}": mean_of(rows, "N512", sets, condition=c)
                           for c in ("shuffled_across", "shuffled_time", "constant_zero", "average_mel")}}
        paired[label] = {f"N512_vs_{ref}": paired_arm_delta(rows, ref, "N512", sets, num_bootstrap=args.bootstrap) for ref in ("A128", "A32")}
        paired[label]["A128_vs_A32"] = paired_arm_delta(rows, "A32", "A128", sets, num_bootstrap=args.bootstrap)

    # Curves: N512 with A128 and A32 for reference.
    keys = [("memorized/correct/mel_l1", "memorized (core)"), ("train_unseen/correct/mel_l1", "unseen crops (core)"),
            ("val/correct/mel_l1", "val (primary; selection)"), ("val2/correct/mel_l1", "val2 (secondary)")]
    fig, axes = plt.subplots(1, len(keys), figsize=(18, 4), sharey=True)
    for ax, (k, title) in zip(axes, keys, strict=True):
        for i, a in enumerate(ARMS):
            h = [e for e in reports[a]["history"] if k in e]
            if h:
                ax.plot([e["step"] for e in h], [e[k] for e in h], label=a, color=f"C{i}")
        best = reports["N512"]["best_checkpoint"]["step"]
        ax.axvline(best, color="C2", ls=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("step")
    axes[0].set_ylabel("mel L1 (correct conditioning)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)

    n = reports["N512"]
    analysis = {
        "n512": {"n_train_tracks": n.get("n_train_tracks"), "best_step": n["best_checkpoint"]["step"],
                 "best_val_mel_l1_at_selection": n["best_checkpoint"]["val_mel_l1"], "final_step": n.get("final_step"),
                 "early_stopping": n.get("early_stopping"), "wandb_run_id": n["wandb_run_id"]},
        "reference_best_steps": {a: reports[a]["best_checkpoint"]["step"] for a in ("A32", "A128")},
        "table": table, "paired": paired,
        "caveat": "Single seed, one run per data size: a scaling data point, not a measurement of MuQ's information ceiling.",
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2))
    md = render(analysis)
    (out / "TASK-20_ANALYSIS.md").write_text(md)
    print(md)
    return 0


def render(a: dict) -> str:
    n = a["n512"]
    lines = [
        "# TASK-20 analysis: N=512 data scaling (recipe A, early stopping)", "",
        (f"N512: {n['n_train_tracks']} training tracks; best step {n['best_step']} (primary val mel L1 {_f(n['best_val_mel_l1_at_selection'])}); "
         f"stopped at step {n['final_step']} ({(n['early_stopping'] or {}).get('reason')}). W&B `{n['wandb_run_id']}`. "
         f"Reference best steps: {a['reference_best_steps']}."), "",
        f"**{a['caveat']}**", "",
        "## Mean mel L1 (correct conditioning, best checkpoint), each set separately", "",
        "| set | what the crops are (for N512) | crops | round trip | A32 | A128 | N512 | N512 final |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, t in a["table"].items():
        lines.append(f"| {label} | {t['meaning']} | {t['n_crops']} | {_f(t['round_trip'])} | {_f(t['A32_best'])} | "
                     f"{_f(t['A128_best'])} | {_f(t['N512_best'])} | {_f(t['N512_final'])} |")
    lines += ["", "## Paired comparisons (same crops; positive = the first arm is better; track-clustered 95% CI)", "",
              "| set | comparison | relative improvement | Δ mel L1 [95% CI] | tracks improved |", "| --- | --- | --- | --- | --- |"]
    for label, ds in a["paired"].items():
        for k, d in ds.items():
            lines.append(f"| {label} | {k} | {100 * d['relative_improvement']:.1f}% | {_f(d['mean_delta'])} "
                         f"[{_f(d['ci_low'])}, {_f(d['ci_high'])}] | {100 * d['proportion_tracks_positive']:.0f}% of {d['n_tracks']} |")
    lines += ["", "## N512 controls (best checkpoint)", "", "| set | correct | shuffled_across | shuffled_time | constant_zero | average_mel |",
              "| --- | --- | --- | --- | --- | --- |"]
    for label, t in a["table"].items():
        lines.append(f"| {label} | {_f(t['N512_best'])} | {_f(t['N512_shuffled_across'])} | {_f(t['N512_shuffled_time'])} | "
                     f"{_f(t['N512_constant_zero'])} | {_f(t['N512_average_mel'])} |")
    lines += ["", "Curves: `curves.png` (dotted line = N512 best step).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
