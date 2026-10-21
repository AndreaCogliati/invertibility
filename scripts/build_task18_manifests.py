"""TASK-18: build the fixed track and crop manifests before any training.

    uv run python scripts/build_task18_manifests.py [--config configs/task18_heldout.yaml] [--verify-audio]

Writes `track_manifest` (48 tracks with split, checksum, resampled length) and `crop_manifest`
(training pool + the four evaluation sets + listening flags) from the committed GTZAN split,
and refuses to overwrite either with different content. `--verify-audio` re-hashes every
selected WAV against the recorded checksum (needs data/raw/gtzan). Deterministic: same config,
same files.
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
    CROP_FIELDS,
    TRACK_FIELDS,
    build_crop_rows,
    check_crop_rows,
    file_sha256,
    resampled_length,
    select_tracks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/task18_heldout.yaml"


def to_csv(rows: list[dict], fields: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows({k: r[k] for k in fields} for r in rows)
    return buf.getvalue()


def write_once(path: Path, text: str) -> str:
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

    with open(REPO_ROOT / cfg["source_manifest"], newline="") as f:
        source = list(csv.DictReader(f))
    raw_root = "data/raw/gtzan"
    selected = select_tracks(source, cfg["track_counts"], cfg["seed"])
    tracks = [{
        "split": r["split"], "track_id": r["track_id"], "genre": r["genre"],
        "path": f"{raw_root}/{r['relative_path']}", "checksum_sha256": r["checksum"],
        "duration_seconds": r["duration_seconds"],
        "resampled_num_samples": resampled_length(int(r["native_num_frames"]), int(r["native_sample_rate"])),
    } for r in selected]
    crops = build_crop_rows(tracks, cfg, cfg["seed"])
    counts = check_crop_rows(crops)

    if args.verify_audio:
        bad = [t["track_id"] for t in tracks if file_sha256(REPO_ROOT / t["path"]) != t["checksum_sha256"]]
        if bad:
            raise SystemExit(f"STOP: checksum mismatch for {bad}")
        print(f"verified sha256 of {len(tracks)} WAVs")

    track_path, crop_path = REPO_ROOT / cfg["track_manifest"], REPO_ROOT / cfg["crop_manifest"]
    status = {str(track_path.relative_to(REPO_ROOT)): write_once(track_path, to_csv(tracks, TRACK_FIELDS)),
              str(crop_path.relative_to(REPO_ROOT)): write_once(crop_path, to_csv(crops, CROP_FIELDS))}
    per_split = collections.Counter(t["split"] for t in tracks)
    genres = {s: dict(sorted(collections.Counter(t["genre"] for t in tracks if t["split"] == s).items()))
              for s in ("train", "val", "test")}
    print(json.dumps({
        "files": status, "tracks_per_split": per_split, "genres_per_split": genres, "crops": counts,
        "listening_examples": sum(int(c["listening"]) for c in crops),
        "track_manifest_sha256": file_sha256(track_path), "crop_manifest_sha256": file_sha256(crop_path),
    }, indent=2))


if __name__ == "__main__":
    main()
