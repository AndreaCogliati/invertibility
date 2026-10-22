"""TASK-20: small blinded listening comparison of A128, N512, and the BigVGAN oracle.

    uv run python scripts/task20_listening_package.py --n512 artifacts/task20/task20_N512_v1 \
        [--a128 artifacts/task19/task19_A128_v1] [--out artifacts/task20/listening_blind_v1]

The 16 fixed TASK-18/19 listening crops (one per primary val/test track). Per example:
`reference_original.wav` plus clip_A..clip_C in random order (oracle, A128 correct, N512 correct).
The key goes to `<out>_KEY/key.json`; do not open it before rating. `ratings.csv`: rate each clip 0-4
("recognizably the same music as the reference") and mark the closer of the two non-oracle-sounding
clips if you can tell.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIPS = ("clip_A", "clip_B", "clip_C")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n512", type=Path, required=True)
    parser.add_argument("--a128", type=Path, default=Path("artifacts/task19/task19_A128_v1"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/task20/listening_blind_v1"))
    args = parser.parse_args()
    dirs = {"A128": REPO_ROOT / args.a128, "N512": REPO_ROOT / args.n512}
    out = (REPO_ROOT / args.out).resolve()
    key_dir = out.with_name(out.name + "_KEY")
    if out.exists() or key_dir.exists():
        raise SystemExit(f"{out} or {key_dir} exists; packages are never overwritten")
    examples = json.loads((dirs["N512"] / "report.json").read_text())["listening"]
    ref_labels = {e["label"] for e in json.loads((dirs["A128"] / "report.json").read_text())["listening"]}
    if {e["label"] for e in examples} != ref_labels:
        raise SystemExit("STOP: listening crops differ between A128 and N512")
    seed = secrets.randbits(64)
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    key, sheet = {"seed": seed, "examples": {}}, []
    for n, i in enumerate(order, 1):
        ex = examples[i]

        def src(arm: str, kind: str, ex: dict = ex) -> Path:
            return dirs[arm] / ex["dir"] / f"{ex['label']}_{kind}.wav"

        for kind in ("original", "bigvgan_oracle"):
            if sha(src("A128", kind)) != sha(src("N512", kind)):
                raise SystemExit(f"{ex['label']} {kind} differs between A128 and N512")
        items = [("bigvgan_oracle", src("N512", "bigvgan_oracle")), ("A128_correct", src("A128", "correct")),
                 ("N512_correct", src("N512", "correct"))]
        rng.shuffle(items)
        d = out / f"example_{n:02d}"
        d.mkdir(parents=True)
        shutil.copyfile(src("N512", "original"), d / "reference_original.wav")
        key["examples"][d.name] = {"label": ex["label"], "eval_set": ex["eval_set"], "track_id": ex["track_id"],
                                   "clips": {c: what for c, (what, _) in zip(CLIPS, items, strict=True)}}
        for c, (_, path) in zip(CLIPS, items, strict=True):
            shutil.copyfile(path, d / f"{c}.wav")
            sheet.append({"example": d.name, "clip": c, "recognizably_same_music_0to4": "", "notes": ""})
        sheet.append({"example": d.name, "clip": "closest_to_reference", "recognizably_same_music_0to4": "", "notes": "A, B, C, or 'same'"})
    with open(out / "ratings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    (out / "README.txt").write_text(
        "Blinded TASK-20 listening. For each example_NN: play reference_original.wav, then clip_A..clip_C.\n"
        "In ratings.csv rate each clip 0-4 (recognizably the same music as the reference). In the\n"
        "'closest_to_reference' row, put the letter of the clip closest to the reference (or 'same').\n"
        "Do not open the _KEY folder until all ratings are done.\n")
    key_dir.mkdir()
    (key_dir / "key.json").write_text(json.dumps(key, indent=2))
    print(f"{len(examples)} examples x 3 clips -> {out.relative_to(REPO_ROOT)}; key in {key_dir.relative_to(REPO_ROOT)} (do not open)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
