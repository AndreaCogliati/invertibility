"""TASK-19: build the fixed track/crop manifests and the GTZAN fault-list report before training.

    uv run python scripts/build_task19_manifests.py [--config configs/task19.yaml] [--verify-audio]

1. Checks the TASK-18 tracks against Sturm's (2013) published fault list
   (data/manifests/gtzan_sturm2013_faults.yaml). If any repetition group spans two TASK-18
   splits, it stops (the TASK-18 manifests are never changed silently).
2. Adds 96 training, 24 validation, and 24 test tracks (genre-balanced, deterministic), none of
   them in any repetition group, distorted, or checksum-duplicated.
3. Checks all 192 tracks again: no repetition group may span two splits.
4. Builds crops with TASK-18's builder, and asserts that every TASK-18 crop row is reproduced
   byte-for-byte (core training tracks, primary val/test).
Writes the track manifest, crop manifest, and fault report once; refuses different content later.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from muq_inversion.task18 import (
    CROP_FIELDS,
    TRACK_FIELDS,
    build_crop_rows,
    check_crop_rows,
    file_sha256,
    resampled_length,
)
from muq_inversion.task19 import extend_selection, fault_report, load_faults

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/task19.yaml"
TRACK19_FIELDS = [*TRACK_FIELDS, "tier", "core", "artist_sturm2013"]
CROP19_FIELDS = [*CROP_FIELDS, "tier", "core"]
RENAME_NEW = {"memorized": "memorized_new", "train_unseen": "train_unseen_new", "val": "val2", "test": "test2"}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_once(path: Path, rows: list[dict], fields: list[str]) -> str:
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows({k: r[k] for k in fields} for r in rows)
    return _write_text_once(path, buf.getvalue())


def _write_text_once(path: Path, text: str) -> str:
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"STOP: {path} exists with different content; manifests are fixed once built")
        return "unchanged"
    path.write_text(text)
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--verify-audio", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    faults = load_faults(REPO_ROOT / cfg["fault_list"])
    source = read_csv(REPO_ROOT / cfg["source_manifest"])
    t18_tracks = read_csv(REPO_ROOT / cfg["task18_track_manifest"])
    t18_crops = read_csv(REPO_ROOT / cfg["task18_crop_manifest"])

    # 1. TASK-18 tracks alone.
    rep18 = fault_report(t18_tracks, faults)
    if rep18["repetition_groups_crossing_splits"] or rep18["distorted"]:
        print(json.dumps(rep18, indent=2))
        raise SystemExit("STOP: TASK-18 tracks are affected by the GTZAN fault list (see above); manifests not changed")

    # 2. New tracks.
    by_id = {r["track_id"]: r for r in source}
    existing = [{**by_id[t["track_id"]], "split": t["split"]} for t in t18_tracks]
    new = extend_selection(source, existing, cfg["track_targets"], faults, cfg["seed"])
    core_ids = {t["track_id"] for t in t18_tracks}
    tracks = [{**t, "tier": "train" if t["split"] == "train" else "primary", "core": "1"} for t in t18_tracks]
    tracks += [{
        "split": r["split"], "track_id": r["track_id"], "genre": r["genre"], "path": f"data/raw/gtzan/{r['relative_path']}",
        "checksum_sha256": r["checksum"], "duration_seconds": r["duration_seconds"],
        "resampled_num_samples": resampled_length(int(r["native_num_frames"]), int(r["native_sample_rate"])),
        "tier": "train" if r["split"] == "train" else "secondary", "core": "0",
    } for r in new]
    for t in tracks:
        t["artist_sturm2013"] = faults["artist"].get(t["track_id"], "")
    order = {"train": 0, "val": 1, "test": 2}
    tracks.sort(key=lambda t: (order[t["split"]], t["core"] != "1", t["track_id"]))

    # 3. All tracks.
    rep = fault_report(tracks, faults)
    if rep["repetition_groups_crossing_splits"]:
        print(json.dumps(rep["repetition_groups_crossing_splits"], indent=2))
        raise SystemExit("STOP: a repetition group spans two TASK-19 splits")

    # 4. Crops: TASK-18 builder; new tracks' eval sets renamed; listening = TASK-18 primary val/test only.
    crops = build_crop_rows(tracks, cfg, cfg["seed"])
    t18_by_id = {r["crop_id"]: r for r in t18_crops}
    listening = {r["crop_id"] for r in t18_crops if r["listening"] == "1" and r["eval_set"] in ("val", "test")}
    for c in crops:
        core = c["track_id"] in core_ids
        c["core"] = "1" if core else "0"
        c["tier"] = next(t["tier"] for t in tracks if t["track_id"] == c["track_id"])
        if not core and c["eval_set"]:
            c["eval_set"] = RENAME_NEW[c["eval_set"]]
        c["listening"] = 1 if c["crop_id"] in listening else 0
        if core:
            ref = t18_by_id.get(c["crop_id"])
            same = ref is not None and all(str(c[k]) == ref[k] for k in CROP_FIELDS if k != "listening")
            if not same:
                raise SystemExit(f"STOP: TASK-18 crop {c['crop_id']} not reproduced exactly")
    missing = set(t18_by_id) - {c["crop_id"] for c in crops}
    if missing:
        raise SystemExit(f"STOP: TASK-18 crops missing from TASK-19: {sorted(missing)[:5]}")
    for c in crops:  # validate geometry with TASK-18's checker (eval-set names mapped back)
        c["_eval_set"] = c["eval_set"]
        c["eval_set"] = {v: k for k, v in RENAME_NEW.items()}.get(c["eval_set"], c["eval_set"])
    check_crop_rows(crops)
    for c in crops:
        c["eval_set"] = c.pop("_eval_set")

    if args.verify_audio:
        bad = [t["track_id"] for t in tracks if file_sha256(REPO_ROOT / t["path"]) != t["checksum_sha256"]]
        if bad:
            raise SystemExit(f"STOP: checksum mismatch for {bad}")
        print(f"verified sha256 of {len(tracks)} WAVs")

    tp, cp, fp = (REPO_ROOT / cfg[k] for k in ("track_manifest", "crop_manifest", "fault_report"))
    status = {str(tp.relative_to(REPO_ROOT)): write_once(tp, tracks, TRACK19_FIELDS),
              str(cp.relative_to(REPO_ROOT)): write_once(cp, crops, CROP19_FIELDS)}
    report = {"task18_tracks": rep18, "task19_tracks": rep}
    status[str(fp.relative_to(REPO_ROOT))] = _write_text_once(fp, json.dumps(report, indent=2, sort_keys=True) + "\n")
    import collections

    summary = {
        "files": status,
        "tracks": {f"{s}/{tier}": n for (s, tier), n in sorted(collections.Counter((t["split"], t["tier"]) for t in tracks).items())},
        "genres": {s: dict(sorted(collections.Counter(t["genre"] for t in tracks if t["split"] == s).items())) for s in order},
        "crops": dict(sorted(collections.Counter(c["eval_set"] or "train_pool" for c in crops).items())),
        "listening_examples": sum(int(c["listening"]) for c in crops),
        "task18_tracks_in_repetition_groups": [m["track_id"] for m in rep18["repetition_members"]],
        "repetition_groups_crossing_splits": len(rep["repetition_groups_crossing_splits"]),
        "heldout_artist_seen_in_128_training": len(rep["heldout_artist_seen_in_training"]),
        "sha256": {k: file_sha256(p) for k, p in (("tracks", tp), ("crops", cp), ("fault_report", fp))},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
