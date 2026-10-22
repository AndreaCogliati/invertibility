"""TASK-20: extend the TASK-19 training set to 512 tracks; keep every held-out track and crop.

    uv run python scripts/build_task20_manifests.py [--config configs/task20.yaml] [--verify-audio]

Adds training tracks from the committed train split (capped genre balance, TASK-19 eligibility),
builds crops with TASK-18's builder, asserts every TASK-19 crop row is reproduced byte-for-byte,
re-runs the Sturm (2013) fault check on all tracks, and writes the manifests once.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
from pathlib import Path

import yaml

from muq_inversion.task18 import (
    build_crop_rows,
    check_crop_rows,
    file_sha256,
    resampled_length,
)
from muq_inversion.task19 import fault_report, load_faults
from muq_inversion.task20 import extend_training_tracks

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/task20.yaml"
RENAME_384 = {"memorized": "memorized_n512", "train_unseen": "train_unseen_n512"}
RENAME_96 = {"memorized": "memorized_new", "train_unseen": "train_unseen_new", "val": "val2", "test": "test2"}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_once(path: Path, text: str) -> str:
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"STOP: {path} exists with different content; manifests are fixed once built")
        return "unchanged"
    path.write_text(text)
    return "written"


def to_csv(rows: list[dict], fields: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows({k: r[k] for k in fields} for r in rows)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--verify-audio", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    base = yaml.safe_load((REPO_ROOT / cfg["base_config"]).read_text())
    cfg = {**base, **cfg}
    faults = load_faults(REPO_ROOT / cfg["fault_list"])
    source = read_csv(REPO_ROOT / cfg["source_manifest"])
    t19_tracks = read_csv(REPO_ROOT / cfg["task19_track_manifest"])
    t19_crops = read_csv(REPO_ROOT / cfg["task19_crop_manifest"])
    track_fields = list(t19_tracks[0]) + ["cohort"]
    crop_fields = list(t19_crops[0]) + ["cohort"]

    by_id = {r["track_id"]: r for r in source}
    existing = [{**by_id[t["track_id"]], "split": t["split"]} for t in t19_tracks]
    new, _ = extend_training_tracks(source, existing, cfg["train_target"], faults, cfg["seed"])
    tracks = [{**t, "cohort": ("core32" if t["core"] == "1" else "new96") if t["split"] == "train" else t["tier"]} for t in t19_tracks]
    tracks += [{
        "split": "train", "track_id": r["track_id"], "genre": r["genre"], "path": f"data/raw/gtzan/{r['relative_path']}",
        "checksum_sha256": r["checksum"], "duration_seconds": r["duration_seconds"],
        "resampled_num_samples": resampled_length(int(r["native_num_frames"]), int(r["native_sample_rate"])),
        "tier": "train", "core": "0", "artist_sturm2013": faults["artist"].get(r["track_id"], ""), "cohort": "new384",
    } for r in sorted(new, key=lambda r: r["track_id"])]

    rep = fault_report(tracks, faults)
    if rep["repetition_groups_crossing_splits"] or rep["distorted"]:
        raise SystemExit(f"STOP: fault list violation: {rep['repetition_groups_crossing_splits']} {rep['distorted']}")
    heldout = {t["track_id"] for t in tracks if t["split"] != "train"}
    train = [t for t in tracks if t["split"] == "train"]
    assert not heldout & {t["track_id"] for t in train}, "held-out track in training"
    assert {t["track_id"] for t in t19_tracks if t["split"] == "train"} <= {t["track_id"] for t in train}, "TASK-19 training track missing"
    assert len({t["track_id"] for t in tracks}) == len(tracks), "duplicate track"

    cohort = {t["track_id"]: t["cohort"] for t in tracks}
    crops = build_crop_rows(tracks, cfg, cfg["seed"])
    t19_by_id = {r["crop_id"]: r for r in t19_crops}
    for c in crops:
        c["cohort"], c["core"] = cohort[c["track_id"]], "1" if cohort[c["track_id"]] in ("core32", "primary") else "0"
        c["tier"] = next(t["tier"] for t in tracks if t["track_id"] == c["track_id"])
        c["_check_eval_set"] = c["eval_set"]
        if c["cohort"] in ("new96", "secondary") and c["eval_set"]:
            c["eval_set"] = RENAME_96[c["eval_set"]]
        elif c["cohort"] == "new384" and c["eval_set"]:
            c["eval_set"] = RENAME_384[c["eval_set"]]
        ref = t19_by_id.get(c["crop_id"])
        c["listening"] = ref["listening"] if ref else 0
        if c["cohort"] != "new384" and (ref is None or any(str(c[k]) != ref[k] for k in ref)):
                raise SystemExit(f"STOP: TASK-19 crop {c['crop_id']} not reproduced exactly")
    if set(t19_by_id) - {c["crop_id"] for c in crops}:
        raise SystemExit("STOP: TASK-19 crops missing")
    for c in crops:  # geometry check with TASK-18 names, then restore
        c["eval_set"], c["_check_eval_set"] = c["_check_eval_set"], c["eval_set"]
    check_crop_rows(crops)
    for c in crops:
        c["eval_set"] = c.pop("_check_eval_set")

    if args.verify_audio:
        bad = [t["track_id"] for t in tracks if file_sha256(REPO_ROOT / t["path"]) != t["checksum_sha256"]]
        if bad:
            raise SystemExit(f"STOP: checksum mismatch for {bad}")
        print(f"verified sha256 of {len(tracks)} WAVs")

    tp, cp, fp = (REPO_ROOT / cfg[k] for k in ("track_manifest", "crop_manifest", "fault_report"))
    status = {str(tp.relative_to(REPO_ROOT)): write_once(tp, to_csv(tracks, track_fields)),
              str(cp.relative_to(REPO_ROOT)): write_once(cp, to_csv(crops, crop_fields)),
              str(fp.relative_to(REPO_ROOT)): write_once(fp, json.dumps(rep, indent=2, sort_keys=True) + "\n")}
    print(json.dumps({
        "files": status, "n_train_tracks": len(train), "target": cfg["train_target"],
        "train_genres": dict(sorted(collections.Counter(t["genre"] for t in train).items())),
        "cohorts": dict(sorted(collections.Counter(t["cohort"] for t in tracks).items())),
        "crops": dict(sorted(collections.Counter(c["eval_set"] or "train_pool" for c in crops).items())),
        "repetition_groups_crossing_splits": len(rep["repetition_groups_crossing_splits"]),
        "heldout_artist_seen_in_training": len(rep["heldout_artist_seen_in_training"]),
        "sha256": {k: file_sha256(p) for k, p in (("tracks", tp), ("crops", cp), ("fault_report", fp))},
    }, indent=2))


if __name__ == "__main__":
    main()
