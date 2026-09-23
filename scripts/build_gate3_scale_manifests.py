"""TASK-05: builds the 5 nested training-set-size manifests for the Gate-3 scaling curve
(2 subset 8 subset 32 subset 128 subset 512), bridging TASK-04's N=2 (perfect memorization) and
TASK-01/02's N~15000 (norm collapse).

## Selection procedure, documented per this task's explicit requirement

1. **Level 1 (N=2) is exactly TASK-04's two excerpts** (`pop.00012`, `disco.00049`), for direct
   continuity with that result -- not a fresh "genre-balanced" pick.
2. **Every subsequent example uses the same crop position, `crop_index=7` (t=14.0s), for every
   track** -- the same position TASK-04 used, already present in
   `data/cache/dense/mel_dac_cache.pt` for all 999 GTZAN tracks (verified: the dense manifest's
   non-overlapping grid puts `crop_index=7` at exactly `t=14.0s` for every track, since
   `crop_start = crop_index * 2.0`). This means every example needed by this task is already
   cached -- no new precompute step, per this task's "do not regenerate features unless
   necessary" instruction.
3. **Additional examples are chosen by round-robin across GTZAN's 10 genres** (alphabetical:
   blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock), each genre
   contributing its train-split tracks in ascending `track_id` order, skipping any track already
   selected at an earlier level (so the nesting property holds by construction: level `k`'s
   selection is always level `k-1`'s selection plus new tracks). This gives balanced genre
   coverage at every level (each level's genre counts differ by at most 1 track from the ideal
   `N/10`), fully deterministic and reproducible from the committed
   `data/manifests/gtzan_manifest_smoke.csv` split.

Train-split genre counts (measured before writing this script): blues 75, classical 80,
country 80, disco 84, hiphop 74, jazz 76, metal 84, pop 79, reggae 84, rock 76 -- all comfortably
above 52 (512/10 rounded up), so every level is reachable without exhausting any genre.

Writes `data/manifests/gate3_scale_{2,8,32,128,512}.csv`, each self-contained with exactly N
rows (not cumulative-diff-only), schema matching `dense_crops.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO_ROOT / "data/manifests/gtzan_manifest_smoke.csv"
DENSE_MANIFEST = REPO_ROOT / "data/manifests/dense_crops.csv"

LEVELS = [2, 8, 32, 128, 512]
CROP_INDEX = 7  # t=14.0s -- same position as TASK-04's two excerpts, already cached
CROP_START_SECONDS = 14.0
CROP_DURATION_SECONDS = 2.0
BASE_TRACKS = ["pop.00012", "disco.00049"]  # exactly TASK-04's two excerpts
GENRE_ORDER = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


def load_train_tracks_by_genre() -> dict[str, list[dict]]:
    with open(SOURCE_MANIFEST, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["corrupted"] != "True" and r["split"] == "train"]
    by_genre: dict[str, list[dict]] = {g: [] for g in GENRE_ORDER}
    for row in rows:
        by_genre[row["genre"]].append(row)
    for tracks in by_genre.values():
        tracks.sort(key=lambda r: r["track_id"])
    return by_genre


def select_nested_track_ids(by_genre: dict[str, list[dict]]) -> dict[int, list[str]]:
    selected: list[str] = list(BASE_TRACKS)
    selected_set = set(selected)
    genre_cursors = {g: 0 for g in GENRE_ORDER}
    level_selections: dict[int, list[str]] = {}

    genre_idx = 0
    for level in LEVELS:
        while len(selected) < level:
            genre = GENRE_ORDER[genre_idx % len(GENRE_ORDER)]
            genre_idx += 1
            tracks = by_genre[genre]
            cursor = genre_cursors[genre]
            while cursor < len(tracks) and tracks[cursor]["track_id"] in selected_set:
                cursor += 1
            if cursor >= len(tracks):
                genre_cursors[genre] = cursor
                continue  # this genre exhausted for now, try the next genre next iteration
            track_id = tracks[cursor]["track_id"]
            selected.append(track_id)
            selected_set.add(track_id)
            genre_cursors[genre] = cursor + 1
        level_selections[level] = list(selected)  # snapshot -- nesting guaranteed by construction
    return level_selections


def main() -> None:
    by_genre = load_train_tracks_by_genre()
    track_meta = {row["track_id"]: row for genre_rows in by_genre.values() for row in genre_rows}
    for track_id in BASE_TRACKS:
        assert track_id in track_meta, f"STOP CONDITION: base track {track_id} not found in train split"

    level_selections = select_nested_track_ids(by_genre)

    for level, track_ids in level_selections.items():
        assert len(track_ids) == level, f"STOP CONDITION: level {level} has {len(track_ids)} tracks, expected {level}"
        assert len(set(track_ids)) == level, f"STOP CONDITION: level {level} has duplicate tracks"

        out_path = REPO_ROOT / f"data/manifests/gate3_scale_{level}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["split", "track_id", "genre", "path", "checksum_sha256", "crop_index", "crop_start_seconds", "crop_duration_seconds"]
            )
            writer.writeheader()
            for track_id in track_ids:
                row = track_meta[track_id]
                writer.writerow(
                    {
                        "split": "train",
                        "track_id": track_id,
                        "genre": row["genre"],
                        "path": f"data/raw/gtzan/{row['relative_path']}",
                        "checksum_sha256": row["checksum"],
                        "crop_index": CROP_INDEX,
                        "crop_start_seconds": f"{CROP_START_SECONDS:.2f}",
                        "crop_duration_seconds": f"{CROP_DURATION_SECONDS:.2f}",
                    }
                )

        from collections import Counter

        genre_counts = Counter(track_meta[t]["genre"] for t in track_ids)
        print(f"wrote {out_path}: {level} tracks, genre counts {dict(sorted(genre_counts.items()))}")


if __name__ == "__main__":
    main()
