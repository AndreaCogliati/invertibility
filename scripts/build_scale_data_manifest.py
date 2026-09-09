"""Builds the deterministic, checksummed, track-disjoint manifest for the data-scaling
experiment (`experiment/muq-to-dac-scale-data`).

Writes `data/manifests/scale_data_crops.csv` (tracked in git, per CLAUDE.md rule 7). Running this
script again reproduces byte-identical output.

## Track selection

100 tracks total (80 train / 10 val / 10 test), all 10 GTZAN genres represented, 10 tracks per
genre: `<genre>.00010` through `<genre>.00019` (deterministic, lowest-numbered block that avoids
the known-corrupted `jazz.00054.wav` entirely). Verified via direct SHA-256 checksum comparison
(see notes/SCRATCHPAD.md) to contain no duplicate-checksum GTZAN defects among these 100 files.

Per genre: 8 train tracks (`.00010`-`.00017`), 1 val track (`.00018`), 1 test track (`.00019`).
80 train + 10 val + 10 test = 100 tracks, no track in more than one split.

## Crops per track

All tracks are ~30s. Deterministic, non-overlapping 2-second crops, spread through the track and
away from the very start/end (fade-in/out risk):

    train tracks: 4 crops, starts 4.0, 10.0, 16.0, 22.0s  -> 80*4 = 320 train crops
    val/test tracks: 2 crops, starts 8.0, 18.0s            -> 10*2 = 20 val, 20 test crops

No guard bands are needed between splits here (unlike Gate 4's Experiment A): every crop's track
belongs to exactly one split, so there is no risk of sample-level leakage between splits -- only
track-level disjointness matters, which is enforced by construction and verified below.
"""

from __future__ import annotations

import csv
from pathlib import Path

from muq_inversion.audio import file_checksum

REPO_ROOT = Path(__file__).resolve().parents[1]
GTZAN_ROOT = REPO_ROOT / "data/raw/gtzan/genres"
MANIFEST_PATH = REPO_ROOT / "data/manifests/scale_data_crops.csv"

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
TRAIN_NUMBERS = list(range(10, 18))  # 8 per genre
VAL_NUMBER = 18
TEST_NUMBER = 19

TRAIN_CROP_STARTS = [4.0, 10.0, 16.0, 22.0]
EVAL_CROP_STARTS = [8.0, 18.0]
CROP_DURATION_SECONDS = 2.0


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    checksum_to_track: dict[str, str] = {}
    track_to_split: dict[str, str] = {}

    for genre in GENRES:
        track_plan = (
            [(n, "train") for n in TRAIN_NUMBERS] + [(VAL_NUMBER, "val")] + [(TEST_NUMBER, "test")]
        )
        for number, split in track_plan:
            track_id = f"{genre}.{number:05d}"
            path = GTZAN_ROOT / genre / f"{track_id}.wav"
            checksum = file_checksum(path)

            prior = checksum_to_track.get(checksum)
            if prior is not None and prior != track_id:
                raise SystemExit(f"duplicate-checksum collision: {track_id} and {prior}")
            checksum_to_track[checksum] = track_id

            prior_split = track_to_split.get(track_id)
            assert prior_split is None, f"{track_id} already assigned to split {prior_split}"
            track_to_split[track_id] = split

            starts = TRAIN_CROP_STARTS if split == "train" else EVAL_CROP_STARTS
            for crop_index, start in enumerate(starts):
                rows.append(
                    {
                        "split": split,
                        "track_id": track_id,
                        "genre": genre,
                        "path": str(path.relative_to(REPO_ROOT)),
                        "checksum_sha256": checksum,
                        "crop_index": str(crop_index),
                        "crop_start_seconds": f"{start:.2f}",
                        "crop_duration_seconds": f"{CROP_DURATION_SECONDS:.2f}",
                    }
                )

    return rows


def main() -> None:
    rows = build_rows()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "track_id",
        "genre",
        "path",
        "checksum_sha256",
        "crop_index",
        "crop_start_seconds",
        "crop_duration_seconds",
    ]
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} crop rows to {MANIFEST_PATH}")

    for split in ("train", "val", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        tracks = sorted({r["track_id"] for r in split_rows})
        print(f"  {split}: {len(split_rows)} crops, {len(tracks)} tracks")


if __name__ == "__main__":
    main()
