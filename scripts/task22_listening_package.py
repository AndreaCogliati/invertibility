"""TASK-22: blinded listening comparison of the first-block (L1) and last_hidden_state (L12) translators.

    uv run python scripts/task22_listening_package.py --l1 artifacts/task22/task22_L1_v1 \
        [--l12 artifacts/task21/task21_N512ext_v1] [--window 200000] [--out artifacts/task22/listening_blind_v1]
    uv run python scripts/task22_listening_package.py --score artifacts/task22/listening_blind_v1   # after rating

The 16 fixed TASK-18..21 listening crops (crop 1 of every primary val/test track: unseen tracks).
Each run's window-best checkpoint (lowest primary val within --window steps, from its own history):
the package stops if a run's evaluated best.pt is not that checkpoint. Per example:
`reference_original.wav` plus clip_A..clip_C in random order (BigVGAN oracle, L1 correct, L12 correct).
The key goes to `<out>_KEY/key.json`; do not open it before rating. Results stay pending until a
human fills in ratings.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import secrets
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from muq_inversion.task22 import window_best

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIPS = ("clip_A", "clip_B", "clip_C")
ORACLE_ATOL = 1e-4


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def score(pkg: Path) -> int:
    key = json.loads((pkg.with_name(pkg.name + "_KEY") / "key.json").read_text())
    with open(pkg / "ratings.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    got: dict[str, dict[str, float]] = {}
    for r in rows:
        v = r["recognizably_same_music_0to4"].strip()
        if r["clip"] in CLIPS and v:
            got.setdefault(r["example"], {})[key["examples"][r["example"]]["clips"][r["clip"]]] = float(v)
    complete = {e: g for e, g in got.items() if len(g) == 3}
    if not complete:
        print("no complete ratings yet: listening results are PENDING")
        return 1
    systems = ("bigvgan_oracle", "L1_correct", "L12_correct")
    out = {"n_examples_rated": len(complete), "n_examples": len(key["examples"]),
           "mean_rating": {s: float(np.mean([g[s] for g in complete.values()])) for s in systems},
           "L1_minus_L12": {"mean": float(np.mean([g["L1_correct"] - g["L12_correct"] for g in complete.values()])),
                            "L1_higher": sum(g["L1_correct"] > g["L12_correct"] for g in complete.values()),
                            "equal": sum(g["L1_correct"] == g["L12_correct"] for g in complete.values()),
                            "L12_higher": sum(g["L1_correct"] < g["L12_correct"] for g in complete.values())},
           "per_example": {e: {"label": key["examples"][e]["label"], **g} for e, g in sorted(complete.items())}}
    (pkg.with_name(pkg.name + "_KEY") / "scores.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_example"}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--l1", type=Path)
    parser.add_argument("--l12", type=Path, default=Path("artifacts/task21/task21_N512ext_v1"))
    parser.add_argument("--window", type=int, default=200000)
    parser.add_argument("--out", type=Path, default=Path("artifacts/task22/listening_blind_v1"))
    parser.add_argument("--score", type=Path, default=None, help="score a rated package instead of building one")
    args = parser.parse_args()
    if args.score:
        return score((REPO_ROOT / args.score).resolve())
    if args.l1 is None:
        raise SystemExit("--l1 is required")
    dirs = {"L1": REPO_ROOT / args.l1, "L12": REPO_ROOT / args.l12}
    reports = {k: json.loads((d / "report.json").read_text()) for k, d in dirs.items()}
    for k, rep in reports.items():
        wb = window_best(rep["history"], args.window)["step"]
        if rep["best_checkpoint"]["step"] != wb:
            raise SystemExit(f"STOP: {k}'s evaluated best.pt (step {rep['best_checkpoint']['step']}) is not its window "
                             f"best (step {wb}); its listening audio is not the window-best checkpoint")
    out = (REPO_ROOT / args.out).resolve()
    key_dir = out.with_name(out.name + "_KEY")
    if out.exists() or key_dir.exists():
        raise SystemExit(f"{out} or {key_dir} exists; packages are never overwritten")
    examples = reports["L1"]["listening"]
    if {e["label"] for e in examples} != {e["label"] for e in reports["L12"]["listening"]}:
        raise SystemExit("STOP: listening crops differ between L1 and L12")
    examples = [e for e in examples if e["eval_set"] in ("val", "test")]  # unseen tracks only
    seed = secrets.randbits(64)
    rng = random.Random(seed)
    rng.shuffle(examples)
    key = {"seed": seed, "window": args.window, "checkpoints": {k: rep["best_checkpoint"] for k, rep in reports.items()},
           "sources": {k: str(d.relative_to(REPO_ROOT)) for k, d in dirs.items()}, "examples": {}}
    sheet = []
    for n, ex in enumerate(examples, 1):
        def src(run: str, kind: str, ex: dict = ex) -> Path:
            return dirs[run] / ex["dir"] / f"{ex['label']}_{kind}.wav"

        if sha(src("L1", "original")) != sha(src("L12", "original")):
            raise SystemExit(f"STOP: {ex['label']} original differs between runs")
        o1, o2 = sf.read(src("L1", "bigvgan_oracle"))[0], sf.read(src("L12", "bigvgan_oracle"))[0]
        oracle_diff = float(np.abs(o1 - o2).max())
        if oracle_diff > ORACLE_ATOL:
            raise SystemExit(f"STOP: {ex['label']} oracle differs between runs by {oracle_diff}")
        items = [("bigvgan_oracle", src("L1", "bigvgan_oracle")), ("L1_correct", src("L1", "correct")),
                 ("L12_correct", src("L12", "correct"))]
        rng.shuffle(items)
        d = out / f"example_{n:02d}"
        d.mkdir(parents=True)
        shutil.copyfile(src("L1", "original"), d / "reference_original.wav")
        key["examples"][d.name] = {"label": ex["label"], "eval_set": ex["eval_set"], "track_id": ex["track_id"],
                                   "genre": ex["genre"], "oracle_max_abs_diff_between_runs": oracle_diff,
                                   "clips": {c: what for c, (what, _) in zip(CLIPS, items, strict=True)}}
        for c, (_, path) in zip(CLIPS, items, strict=True):
            shutil.copyfile(path, d / f"{c}.wav")
            sheet.append({"example": d.name, "clip": c, "recognizably_same_music_0to4": "", "notes": ""})
    with open(out / "ratings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    (out / "README.txt").write_text(
        "Blinded TASK-22 listening. For each example_NN: play reference_original.wav, then clip_A..clip_C.\n"
        "In ratings.csv rate each clip 0-4: how recognizably it is the same music as the reference\n"
        "(0 = unrelated, 4 = clearly the same music). Use the notes column for what survives or is lost\n"
        "(rhythm, pitch/harmony, instrumentation, dynamics, timbre, artifacts).\n"
        "Do not open the _KEY folder until all ratings are done.\n")
    key_dir.mkdir()
    (key_dir / "key.json").write_text(json.dumps(key, indent=2))
    print(f"{len(examples)} examples x 3 clips -> {out.relative_to(REPO_ROOT)}; key in {key_dir.relative_to(REPO_ROOT)} (do not open)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
