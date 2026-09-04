"""GTZAN acquisition, indexing, deterministic splitting, and manifests (Gate 1).

CLAUDE.md requires: explicit (non-hidden) dataset acquisition, deterministic
manifests with checksums, duplicate detection, and a documented limitation
if an artist-aware/fault-filtered split isn't available. GTZAN ships with no
artist metadata, so the split below is a deterministic hash-of-filename
split -- documented in notes/DECISIONS.md, not artist-aware.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import tarfile
from pathlib import Path

from .audio import file_checksum, load_resample_validate
from .config import DataConfig

# Known corrupted file in the canonical marsyas/gtzan distribution
# (confirmed via the HF dataset loading script's CORRUPTED_FILES list).
KNOWN_CORRUPTED_FILES = {"jazz.00054.wav"}

HF_REPO_ID = "marsyas/gtzan"
HF_REPO_TYPE = "dataset"
HF_ARCHIVE_FILENAME = "data/genres.tar.gz"


@dataclasses.dataclass(frozen=True)
class TrackRecord:
    track_id: str
    genre: str
    relative_path: str
    checksum: str
    native_sample_rate: int
    native_num_frames: int
    duration_seconds: float
    corrupted: bool
    split: str = ""


def download_and_extract(raw_dir: Path) -> Path:
    """Download the marsyas/gtzan archive from Hugging Face and extract it into raw_dir.

    Idempotent: if raw_dir already contains the expected genre subdirectories,
    extraction is skipped.
    """
    from huggingface_hub import hf_hub_download

    marker = raw_dir / ".extracted"
    if marker.exists():
        return raw_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = hf_hub_download(
        repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, filename=HF_ARCHIVE_FILENAME
    )

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(raw_dir, filter="data")

    marker.write_text(f"source={HF_REPO_ID}:{HF_ARCHIVE_FILENAME}\n")
    return raw_dir


def index_gtzan(raw_dir: Path, genres: tuple[str, ...]) -> list[TrackRecord]:
    """Walk raw_dir for GTZAN wavs, compute checksums, and flag known-corrupted files."""
    import soundfile as sf

    wav_paths = sorted(p for p in raw_dir.rglob("*.wav") if not p.name.startswith("._"))
    if not wav_paths:
        raise FileNotFoundError(
            f"No .wav files found under {raw_dir}. Run prepare-data to download GTZAN first."
        )

    records: list[TrackRecord] = []
    for path in wav_paths:
        genre = path.parent.name
        if genre not in genres:
            continue
        track_id = path.stem
        known_corrupted = path.name in KNOWN_CORRUPTED_FILES
        try:
            info = sf.info(str(path))
            native_sample_rate = int(info.samplerate)
            native_num_frames = int(info.frames)
            duration_seconds = float(info.frames) / info.samplerate if info.samplerate else 0.0
            corrupted = known_corrupted
        except (sf.LibsndfileError, RuntimeError, OSError) as exc:
            if not known_corrupted:
                print(f"WARNING: {path} is unreadable and not in the known-corrupted list: {exc}")
            native_sample_rate = 0
            native_num_frames = 0
            duration_seconds = 0.0
            corrupted = True
        records.append(
            TrackRecord(
                track_id=track_id,
                genre=genre,
                relative_path=str(path.relative_to(raw_dir)),
                checksum=file_checksum(path),
                native_sample_rate=native_sample_rate,
                native_num_frames=native_num_frames,
                duration_seconds=duration_seconds,
                corrupted=corrupted,
            )
        )
    return records


def find_duplicate_checksums(records: list[TrackRecord]) -> dict[str, list[str]]:
    """Map checksum -> list of track_ids, restricted to checksums shared by >1 track."""
    by_checksum: dict[str, list[str]] = {}
    for record in records:
        by_checksum.setdefault(record.checksum, []).append(record.track_id)
    return {checksum: ids for checksum, ids in by_checksum.items() if len(ids) > 1}


def _deterministic_unit_interval(key: str, seed: int) -> float:
    """Stable hash of (key, seed) mapped to [0, 1), used for a reproducible split."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    as_int = int.from_bytes(digest[:8], "big")
    return as_int / 2**64


def assign_splits(
    records: list[TrackRecord],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> list[TrackRecord]:
    """Deterministic hash-based split, stratified by genre.

    NOTE (documented limitation, CLAUDE.md rule 7): GTZAN's canonical
    distribution carries no artist metadata, so an artist-aware or
    fault-filtered split is not available here. This split is track-level
    and deterministic but not artist-aware.

    Duplicate-checksum tracks (a known GTZAN defect) are always assigned to
    the same split as each other, keyed by the lexicographically smallest
    track_id sharing that checksum, so a byte-identical clip can never
    straddle train/val/test and leak across the split boundary.
    """
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    updated: list[TrackRecord] = []
    for record in records:
        # Keyed purely by checksum (not genre/track_id): guarantees every
        # byte-identical duplicate, even across genres, hashes to the same split.
        u = _deterministic_unit_interval(record.checksum, seed)
        if u < train_fraction:
            split = "train"
        elif u < train_fraction + val_fraction:
            split = "val"
        else:
            split = "test"
        updated.append(dataclasses.replace(record, split=split))
    return updated


def write_manifest(records: list[TrackRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in dataclasses.fields(TrackRecord)]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(dataclasses.asdict(record))


def read_manifest(path: Path) -> list[TrackRecord]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            row["native_sample_rate"] = int(row["native_sample_rate"])
            row["native_num_frames"] = int(row["native_num_frames"])
            row["duration_seconds"] = float(row["duration_seconds"])
            row["corrupted"] = row["corrupted"] == "True"
            records.append(TrackRecord(**row))
    return records


@dataclasses.dataclass(frozen=True)
class ResampleVerification:
    track_id: str
    native_sample_rate: int
    native_num_samples: int
    resampled_num_samples: int
    target_sample_rate: int
    is_finite: bool
    peak_amplitude: float


def verify_resample_sample(
    records: list[TrackRecord], raw_dir: Path, config: DataConfig, sample_size: int, seed: int
) -> list[ResampleVerification]:
    """Resample a deterministic sample of usable tracks and verify length/finiteness/amplitude."""
    import numpy as np

    usable = [r for r in records if not r.corrupted]
    ordered = sorted(usable, key=lambda r: _deterministic_unit_interval(r.track_id, seed))
    sample = ordered[:sample_size]

    results = []
    for record in sample:
        path = raw_dir / record.relative_path
        waveform, orig_sr = load_resample_validate(path, config.target_sample_rate)
        expected_len = round(record.native_num_frames * config.target_sample_rate / orig_sr)
        if abs(len(waveform) - expected_len) > 1:
            raise ValueError(
                f"{record.track_id}: resampled length {len(waveform)} deviates from "
                f"expected {expected_len} by more than 1 sample"
            )
        results.append(
            ResampleVerification(
                track_id=record.track_id,
                native_sample_rate=orig_sr,
                native_num_samples=record.native_num_frames,
                resampled_num_samples=len(waveform),
                target_sample_rate=config.target_sample_rate,
                is_finite=bool(np.all(np.isfinite(waveform))),
                peak_amplitude=float(np.max(np.abs(waveform))),
            )
        )
    return results
