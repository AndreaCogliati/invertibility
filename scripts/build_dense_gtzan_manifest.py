"""Builds the dense-coverage crop manifest for `tasks/TASK-02.md` (Option B: non-overlapping
2.0s crops tiling each track), reusing Gate 1's committed full-GTZAN split unchanged -- the exact
same split `experiment/muq-to-dac-more-data` (and `tasks/TASK-01.md`) used.

Verified before writing any code that `data/manifests/gtzan_manifest_smoke.csv`'s `split` column
matches `more_data_crops.csv`'s track->split assignment for all 999 non-corrupted tracks (0
mismatches) -- this manifest is Gate 1's own committed split, not a re-derivation.

## Option B chosen over A/C (documented per this task's "document the rationale" requirement)

Option A (50% overlapping crops) would yield ~29 crops/track but with ~2x redundant audio
content per crop (each sample appears in ~2 crops) -- more gradient steps per second of audio,
not more distinct audio. Option C (5s/10s crops) changes the crop length itself, which nothing in
this task asks for and which would also change `TemporalConvNet`'s effective exposure per step.
**Option B (non-overlapping 2.0s crops)** keeps the crop length identical to TASK-01 (so the
representation/architecture/frame-alignment code needs zero changes), yields ~100% track coverage
with no duplicated audio, and is the simplest of the three -- exactly this task's own selection
criterion ("choose the simplest option that maximizes audio coverage").

## Measured dataset size (before implementation, per this task's explicit requirement)

Track durations range 29.93-30.65s (`gtzan_manifest_smoke.csv`'s `duration_seconds`, measured at
Gate 1), giving 14 or 15 non-overlapping 2.0s crops per track. Summed over all 999 non-corrupted
tracks, split by Gate 1's committed assignment: **11873 train / 1409 val / 1694 test = 14976
total crops** (vs. TASK-01's 1584/188/226 = 1998 -- a 7.50x/7.50x/7.50x increase, ~7.50x overall).
14976 crops x 2.0s = 29952s (99.94% of GTZAN's ~29970s total, vs. TASK-01's 13.3%).

Storage estimate (before running the precompute script, per this task's explicit requirement):
DAC target (1, 150, 1024) float32 = 614400 bytes/crop -> ~9.20GB for 14976 crops; log-mel (1,
150, 128) float32 = 76800 bytes/crop -> ~1.15GB. Combined ~10.35GB, git-ignored, comfortably
within this machine's 689GB free disk (measured via `df -h`, 2026-09-14).

Writes `data/manifests/dense_crops.csv`, same schema as `more_data_crops.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO_ROOT / "data/manifests/gtzan_manifest_smoke.csv"  # Gate 1's committed split
OUTPUT_MANIFEST = REPO_ROOT / "data/manifests/dense_crops.csv"

CROP_DURATION_SECONDS = 2.0


def main() -> None:
    with open(SOURCE_MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    crops_by_split: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    tracks_by_split: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        if row["corrupted"] == "True":
            continue
        duration = float(row["duration_seconds"])
        num_crops = int(duration // CROP_DURATION_SECONDS)
        assert num_crops > 0, f"STOP CONDITION: {row['track_id']} too short ({duration}s) for even one crop"
        split = row["split"]
        tracks_by_split[split] += 1
        for crop_index in range(num_crops):
            crop_start = crop_index * CROP_DURATION_SECONDS
            assert crop_start + CROP_DURATION_SECONDS <= duration + 1e-6, (
                f"STOP CONDITION: crop [{crop_start},{crop_start + CROP_DURATION_SECONDS}) "
                f"exceeds {row['track_id']}'s duration {duration}"
            )
            out_rows.append(
                {
                    "split": split,
                    "track_id": row["track_id"],
                    "genre": row["genre"],
                    "path": f"data/raw/gtzan/{row['relative_path']}",
                    "checksum_sha256": row["checksum"],
                    "crop_index": crop_index,
                    "crop_start_seconds": f"{crop_start:.2f}",
                    "crop_duration_seconds": f"{CROP_DURATION_SECONDS:.2f}",
                }
            )
            crops_by_split[split] += 1

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "track_id",
                "genre",
                "path",
                "checksum_sha256",
                "crop_index",
                "crop_start_seconds",
                "crop_duration_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    total_crops = sum(crops_by_split.values())
    total_tracks = sum(tracks_by_split.values())
    print(f"wrote {OUTPUT_MANIFEST}: {total_crops} crops, {total_tracks} tracks")
    for split in ("train", "val", "test"):
        print(f"  {split}: {crops_by_split[split]} crops, {tracks_by_split[split]} tracks")
    print(f"total audio duration: {total_crops * CROP_DURATION_SECONDS:.1f}s "
          f"({total_crops * CROP_DURATION_SECONDS / 60:.1f} min)")


if __name__ == "__main__":
    main()
