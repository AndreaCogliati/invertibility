"""Builds the deterministic, checksummed crop manifest for the Gate 4 generalization experiment.

Writes `data/manifests/gate4_crops.csv` (tracked in git, per CLAUDE.md rule 7: "save the exact
manifests"). Running this script again reproduces byte-identical output (pure function of the
constants below plus each file's on-disk checksum) -- not idempotent in the sense of skipping
work, but deterministic in the sense of never silently drifting.

## Experiment A: unseen crops from the two tracks used throughout Gate 3

Both tracks are 30.00s. Crop layout (seconds), with a 1.0s guard band between every region so no
window -- including the guard itself -- ever overlaps an adjacent region:

    train:  8 crops, starts 0,2,4,...,14   -> covers [0, 16)
    guard:                                     [16, 17)
    val:    2 crops, starts 17, 19          -> covers [17, 21)
    guard:                                     [21, 22)
    test:   2 crops, starts 22, 24          -> covers [22, 26)
    (unused tail: [26, 30), not assigned to any split)

## Experiment B: eight track-disjoint GTZAN tracks across four genres

Chosen deterministically (lowest available track numbers not already used by Experiment A, not
the known-corrupted `jazz.00054.wav`, verified via direct checksum comparison to have no
duplicate-checksum collisions among each other or with Experiment A's two tracks):

    train (8): pop.00001, pop.00002, disco.00001, disco.00002,
               classical.00001, classical.00002, rock.00001, rock.00002
    val   (2): pop.00003, disco.00003
    test  (2): classical.00003, rock.00003

Train tracks get 2 crops each (starts 5.0s, 15.0s -- arbitrary but deterministic, away from
track start/end where fades are more likely); val/test tracks get 1 fixed crop each (start
10.0s). No guard bands needed here since these are entirely different audio files -- track-level
disjointness, not sample-level adjacency, is what matters.
"""

from __future__ import annotations

import csv
from itertools import pairwise
from pathlib import Path

from muq_inversion.audio import file_checksum

REPO_ROOT = Path(__file__).resolve().parents[1]
GTZAN_ROOT = REPO_ROOT / "data/raw/gtzan/genres"
MANIFEST_PATH = REPO_ROOT / "data/manifests/gate4_crops.csv"

CROP_DURATION_SECONDS = 2.0

# Experiment A: (split, [crop_start_seconds, ...])
EXPERIMENT_A_TRACKS = [
    ("pop", "00012"),
    ("disco", "00049"),
]
EXPERIMENT_A_SPLITS = [
    ("train", [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]),
    ("val", [17.0, 19.0]),
    ("test", [22.0, 24.0]),
]

# Experiment B: (split, genre, track_number, [crop_start_seconds, ...])
EXPERIMENT_B_TRACKS = [
    ("train", "pop", "00001", [5.0, 15.0]),
    ("train", "pop", "00002", [5.0, 15.0]),
    ("train", "disco", "00001", [5.0, 15.0]),
    ("train", "disco", "00002", [5.0, 15.0]),
    ("train", "classical", "00001", [5.0, 15.0]),
    ("train", "classical", "00002", [5.0, 15.0]),
    ("train", "rock", "00001", [5.0, 15.0]),
    ("train", "rock", "00002", [5.0, 15.0]),
    ("val", "pop", "00003", [10.0]),
    ("val", "disco", "00003", [10.0]),
    ("test", "classical", "00003", [10.0]),
    ("test", "rock", "00003", [10.0]),
]


def _verify_experiment_a_regions_disjoint() -> None:
    """Assert no split's crop windows (each `CROP_DURATION_SECONDS` wide) can overlap another
    split's, with the 1.0s guard bands accounted for -- a stop-condition check per this
    experiment's own spec ("stop and diagnose ... if any split overlaps at the ... sample
    level"), run at manifest-build time rather than trusted from the docstring alone."""
    intervals: list[tuple[str, float, float]] = []
    for split, starts in EXPERIMENT_A_SPLITS:
        for start in starts:
            intervals.append((split, start, start + CROP_DURATION_SECONDS))
    intervals.sort(key=lambda t: t[1])
    for (split_a, _, end_a), (split_b, start_b, _) in pairwise(intervals):
        gap = start_b - end_a
        assert gap >= 0, f"overlap detected between {split_a} and {split_b}: gap={gap}"
        if split_a != split_b:
            assert gap >= 1.0, f"guard interval too small between {split_a} and {split_b}: {gap}s"


def build_rows() -> list[dict[str, str]]:
    _verify_experiment_a_regions_disjoint()
    rows: list[dict[str, str]] = []

    for genre, number in EXPERIMENT_A_TRACKS:
        track_id = f"{genre}.{number}"
        path = GTZAN_ROOT / genre / f"{track_id}.wav"
        checksum = file_checksum(path)
        for split, starts in EXPERIMENT_A_SPLITS:
            for crop_index, start in enumerate(starts):
                rows.append(
                    {
                        "experiment": "A",
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

    for split, genre, number, starts in EXPERIMENT_B_TRACKS:
        track_id = f"{genre}.{number}"
        path = GTZAN_ROOT / genre / f"{track_id}.wav"
        checksum = file_checksum(path)
        for crop_index, start in enumerate(starts):
            rows.append(
                {
                    "experiment": "B",
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

    # Verify no track appears in more than one Experiment B split, and no duplicate checksums
    # across the whole manifest (stop-condition: "any split overlaps at the track ... level").
    track_to_splits: dict[str, set[str]] = {}
    checksum_to_track: dict[str, str] = {}
    for row in rows:
        if row["experiment"] == "B":
            track_to_splits.setdefault(row["track_id"], set()).add(row["split"])
        prior = checksum_to_track.get(row["checksum_sha256"])
        if prior is not None and prior != row["track_id"]:
            raise SystemExit(
                f"duplicate-checksum collision: {row['track_id']} and {prior} are byte-identical"
            )
        checksum_to_track[row["checksum_sha256"]] = row["track_id"]
    for track_id, splits in track_to_splits.items():
        assert len(splits) == 1, f"{track_id} appears in multiple Experiment B splits: {splits}"

    return rows


def main() -> None:
    rows = build_rows()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} crop rows to {MANIFEST_PATH}")

    for experiment in ("A", "B"):
        for split in ("train", "val", "test"):
            count = sum(1 for r in rows if r["experiment"] == experiment and r["split"] == split)
            tracks = sorted({r["track_id"] for r in rows if r["experiment"] == experiment and r["split"] == split})
            print(f"  Experiment {experiment} / {split}: {count} crops, tracks={tracks}")


if __name__ == "__main__":
    main()
