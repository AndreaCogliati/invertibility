"""Builds the deterministic, checksummed, track-disjoint manifest for the further data-scaling
experiment (`experiment/muq-to-dac-more-data`) -- this time using essentially the entire GTZAN
dataset, not another hand-picked subset.

Writes `data/manifests/more_data_crops.csv` (tracked in git, per CLAUDE.md rule 7).

## Track selection: reusing Gate 1's own committed manifest, not re-deriving a split

`data/manifests/gtzan_manifest_smoke.csv` (committed during Gate 1, 2026-09-04) already contains
a full 1000-track, checksummed, deterministic, duplicate-safe train/val/test split: 793 train /
94 val / 113 test, with the one known-corrupted file (`jazz.00054.wav`) flagged, and all 14
duplicate-checksum GTZAN defect pairs already kept within a single split each (`assign_splits`
hashes on checksum specifically to guarantee this -- see notes/DECISIONS.md
#gtzan-source-and-defects). Reusing this directly is both more thorough (the full dataset, not
another ~100-600 track hand-picked subset) and safer (leakage-checked machinery that already
exists and is documented) than re-deriving a new split for this experiment. The only
transformation applied here is dropping the corrupted row and adding deterministic crop
boundaries -- no new split logic.

This gives 793 training tracks (> 500 target), 94 validation tracks, 113 test tracks -- track
counts are the underlying manifest's own natural split proportions (roughly 80/9/11%), not
independently chosen for this experiment.

## Crops per track

2 deterministic 2-second crops per track (starts 5.0s, 15.0s -- same convention as
`experiment/muq-to-dac-scale-data`'s training crops), applied uniformly to all three splits:

    793 train tracks x 2 = 1586 train crops
    94 val tracks x 2     = 188 val crops
    113 test tracks x 2   = 226 test crops
    Total: 2000 crops

Kept to 2 crops/track (not 4, as `experiment/muq-to-dac-scale-data`'s 80-track experiment used)
specifically because the track count itself is already ~10x larger than that experiment -- crop
diversity per track matters much less than track diversity for testing track-level
generalization, and this keeps the total crop count (and therefore precompute/training time)
laptop-friendly.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO_ROOT / "data/manifests/gtzan_manifest_smoke.csv"
MANIFEST_PATH = REPO_ROOT / "data/manifests/more_data_crops.csv"

CROP_STARTS = [5.0, 15.0]
CROP_DURATION_SECONDS = 2.0


def build_rows() -> list[dict[str, str]]:
    with open(SOURCE_MANIFEST, newline="") as f:
        source_rows = list(csv.DictReader(f))

    usable = [r for r in source_rows if r["corrupted"] != "True"]
    assert len(usable) == len(source_rows) - 1, (
        f"expected exactly 1 corrupted row dropped, dropped {len(source_rows) - len(usable)}"
    )

    # Re-verify track-disjointness and no cross-split duplicate-checksum leakage here, not just
    # trust the source manifest's own documentation -- cheap and directly relevant to this
    # experiment's "no leakage" requirement.
    checksum_to_split: dict[str, str] = {}
    for row in usable:
        prior_split = checksum_to_split.get(row["checksum"])
        if prior_split is not None:
            assert prior_split == row["split"], (
                f"STOP CONDITION: checksum {row['checksum'][:12]} spans splits "
                f"{prior_split} and {row['split']}"
            )
        checksum_to_split[row["checksum"]] = row["split"]

    rows: list[dict[str, str]] = []
    for row in usable:
        for crop_index, start in enumerate(CROP_STARTS):
            rows.append(
                {
                    "split": row["split"],
                    "track_id": row["track_id"],
                    "genre": row["genre"],
                    "path": f"data/raw/gtzan/{row['relative_path']}",
                    "checksum_sha256": row["checksum"],
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
        tracks = {r["track_id"] for r in split_rows}
        genres = sorted({r["genre"] for r in split_rows})
        print(f"  {split}: {len(split_rows)} crops, {len(tracks)} tracks, genres={genres}")


if __name__ == "__main__":
    main()
