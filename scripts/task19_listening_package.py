"""TASK-19: blinded listening package from the four arms' listening exports.

    uv run python scripts/task19_listening_package.py \
        --arm A32=artifacts/task19/task19_A32_v1 --arm A128=... --arm B32=... --arm B128=... \
        [--out artifacts/task19/listening_blind_v1]

For each of the 16 fixed listening crops (identical in every arm): `reference_original.wav` plus 8
clips with random names (clip_A..clip_H) in random order: BigVGAN oracle, each arm's correct
prediction, and A32's shuffled_across, shuffled_time, and constant_zero controls. The mapping is
written to `<out>_KEY/key.json`, a separate folder: do not open it before rating. `ratings.csv` is
the answer sheet. Originals and oracles must be identical across arms (checked by hash).
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

from muq_inversion.task19 import ARMS

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIP_NAMES = [f"clip_{c}" for c in "ABCDEFGH"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, help="ARM=artifact_dir, once per arm")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/task19/listening_blind_v1")
    args = parser.parse_args()
    args.out = (REPO_ROOT / args.out).resolve()
    dirs = {k: REPO_ROOT / v for k, v in (a.split("=", 1) for a in args.arm)}
    if set(dirs) != set(ARMS):
        raise SystemExit(f"need exactly the arms {ARMS}")
    key_dir = args.out.with_name(args.out.name + "_KEY")
    if args.out.exists() or key_dir.exists():
        raise SystemExit(f"{args.out} or {key_dir} exists; packages are never overwritten")
    examples = json.loads((dirs["A32"] / "report.json").read_text())["listening"]
    seed = secrets.randbits(64)
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    key = {"seed": seed, "examples": {}}
    sheet = []
    for n, i in enumerate(order, 1):
        ex = examples[i]
        def src(arm: str, kind: str, ex: dict = ex) -> Path:
            return dirs[arm] / ex["dir"] / f"{ex['label']}_{kind}.wav"

        for kind in ("original", "bigvgan_oracle"):
            hashes = {arm: sha(src(arm, kind)) for arm in ARMS}
            if len(set(hashes.values())) != 1:
                raise SystemExit(f"{ex['label']} {kind} differs across arms: {hashes}")
        items = [("bigvgan_oracle", src("A32", "bigvgan_oracle"))]
        items += [(f"{arm}_correct", src(arm, "correct")) for arm in ARMS]
        items += [(f"A32_{c}", src("A32", c)) for c in ("shuffled_across", "shuffled_time", "constant_zero")]
        rng.shuffle(items)
        d = args.out / f"example_{n:02d}"
        d.mkdir(parents=True)
        shutil.copyfile(src("A32", "original"), d / "reference_original.wav")
        key["examples"][d.name] = {"label": ex["label"], "eval_set": ex["eval_set"], "track_id": ex["track_id"],
                                   "clips": {name: what for name, (what, _) in zip(CLIP_NAMES, items, strict=True)}}
        for name, (_, path) in zip(CLIP_NAMES, items, strict=True):
            shutil.copyfile(path, d / f"{name}.wav")
            sheet.append({"example": d.name, "clip": name, "recognizably_same_music_0to4": "", "closer_than_others_rank": "", "notes": ""})
    with open(args.out / "ratings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    (args.out / "README.txt").write_text(
        "Blinded TASK-19 listening. For each example_NN: play reference_original.wav, then clip_A..clip_H.\n"
        "Rate each clip in ratings.csv: recognizably the same music as the reference (0 = no .. 4 = clearly),\n"
        "and optionally rank the clips by closeness. Do not open the _KEY folder until all ratings are done.\n")
    key_dir.mkdir()
    (key_dir / "key.json").write_text(json.dumps(key, indent=2))
    print(f"{len(examples)} examples x 8 clips -> {args.out.relative_to(REPO_ROOT)}; key in {key_dir.relative_to(REPO_ROOT)} (do not open)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
