"""TASK-14: builds the fixed, versioned held-out (validation + test) crop manifest for the full
GTZAN Gate 5 pilot.

Per this task's explicit instructions:

- Training crops are NOT drawn from a fixed manifest -- they are sampled on the fly, broadly
  across the full 792-track training split, by `gate5_common.TrainCropSampler` (a genuinely
  random-start-offset sampler, not a rotation over a small fixed set). No file is built for them.
- Validation and test crops MUST be a fixed, versioned manifest, built once before training,
  reused unchanged for every checkpoint and every conditioning control.

This script derives that manifest from `data/manifests/more_data_crops.csv` (itself built from
`data/manifests/gtzan_manifest_smoke.csv`'s committed 792/94/113 track split, 2 fixed 2.0s crops
per track -- see `scripts/build_more_data_manifest.py`) by keeping only its `val`/`test` rows --
no new crop-selection decision is made here; this reuses the project's own established "two fixed
2.0s windows per track" convention (documented in `notes/DECISIONS.md#spectrogram-positive-control`)
rather than inventing a new one.

The output uses the exact same column schema as `data/manifests/gate4_crops.csv`
(`experiment,split,track_id,genre,path,checksum_sha256,crop_index,crop_start_seconds,
crop_duration_seconds`) with `experiment` fixed to `"gate5"` for every row, so
`gate4_encodec_common.py`'s `load_gate4_manifest`/`load_gate4_examples`/`build_conditioning`
(already validated by 14 tests in TASK-13) work on it completely unchanged -- no new manifest
loader or conditioning-control code is needed for TASK-14's held-out evaluation.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO_ROOT / "data/manifests/more_data_crops.csv"
OUTPUT_MANIFEST = REPO_ROOT / "data/manifests/gate5_heldout_crops.csv"
EXPERIMENT_LABEL = "gate5"

FIELDNAMES = [
    "experiment",
    "split",
    "track_id",
    "genre",
    "path",
    "checksum_sha256",
    "crop_index",
    "crop_start_seconds",
    "crop_duration_seconds",
]


def main() -> None:
    with open(SOURCE_MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))

    heldout_rows = [row for row in rows if row["split"] in ("val", "test")]
    if not heldout_rows:
        raise SystemExit(f"STOP CONDITION: no val/test rows found in {SOURCE_MANIFEST}")

    val_tracks = {r["track_id"] for r in heldout_rows if r["split"] == "val"}
    test_tracks = {r["track_id"] for r in heldout_rows if r["split"] == "test"}
    overlap = val_tracks & test_tracks
    if overlap:
        raise SystemExit(f"STOP CONDITION: tracks in both val and test: {overlap}")

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in heldout_rows:
            writer.writerow(
                {
                    "experiment": EXPERIMENT_LABEL,
                    "split": row["split"],
                    "track_id": row["track_id"],
                    "genre": row["genre"],
                    "path": row["path"],
                    "checksum_sha256": row["checksum_sha256"],
                    "crop_index": row["crop_index"],
                    "crop_start_seconds": row["crop_start_seconds"],
                    "crop_duration_seconds": row["crop_duration_seconds"],
                }
            )

    val_rows = [r for r in heldout_rows if r["split"] == "val"]
    test_rows = [r for r in heldout_rows if r["split"] == "test"]
    print(
        f"wrote {len(heldout_rows)} rows to {OUTPUT_MANIFEST} "
        f"(val: {len(val_rows)} crops / {len(val_tracks)} tracks, "
        f"test: {len(test_rows)} crops / {len(test_tracks)} tracks)"
    )


if __name__ == "__main__":
    main()
