import dataclasses

from muq_inversion.data import (
    TrackRecord,
    assign_splits,
    find_duplicate_checksums,
    read_manifest,
    write_manifest,
)


def _make_record(track_id: str, genre: str, checksum: str) -> TrackRecord:
    return TrackRecord(
        track_id=track_id,
        genre=genre,
        relative_path=f"{genre}/{track_id}.wav",
        checksum=checksum,
        native_sample_rate=22050,
        native_num_frames=661794,
        duration_seconds=30.0,
        corrupted=False,
    )


def test_assign_splits_is_deterministic():
    records = [_make_record(f"blues.{i:05d}", "blues", f"checksum{i}") for i in range(100)]
    split_a = assign_splits(records, 0.8, 0.1, 0.1, seed=1234)
    split_b = assign_splits(records, 0.8, 0.1, 0.1, seed=1234)
    assert [r.split for r in split_a] == [r.split for r in split_b]


def test_assign_splits_respects_fractions_approximately():
    records = [_make_record(f"blues.{i:05d}", "blues", f"checksum{i}") for i in range(1000)]
    split = assign_splits(records, 0.8, 0.1, 0.1, seed=1234)
    counts = {"train": 0, "val": 0, "test": 0}
    for r in split:
        counts[r.split] += 1
    assert 700 < counts["train"] < 900
    assert 50 < counts["val"] < 150
    assert 50 < counts["test"] < 150


def test_assign_splits_keeps_duplicate_checksums_together():
    records = [
        _make_record("metal.00058", "metal", "shared_checksum"),
        _make_record("rock.00016", "rock", "shared_checksum"),
    ] + [_make_record(f"blues.{i:05d}", "blues", f"checksum{i}") for i in range(50)]
    split = assign_splits(records, 0.8, 0.1, 0.1, seed=1234)
    splits_by_id = {r.track_id: r.split for r in split}
    assert splits_by_id["metal.00058"] == splits_by_id["rock.00016"]


def test_find_duplicate_checksums():
    records = [
        _make_record("a", "blues", "checksum1"),
        _make_record("b", "blues", "checksum1"),
        _make_record("c", "blues", "checksum2"),
    ]
    duplicates = find_duplicate_checksums(records)
    assert duplicates == {"checksum1": ["a", "b"]}


def test_manifest_round_trip(tmp_path):
    records = [_make_record(f"blues.{i:05d}", "blues", f"checksum{i}") for i in range(5)]
    records = assign_splits(records, 0.8, 0.1, 0.1, seed=1234)
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(records, manifest_path)
    loaded = read_manifest(manifest_path)
    assert [dataclasses.asdict(r) for r in loaded] == [dataclasses.asdict(r) for r in records]
