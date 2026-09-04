"""Deterministic audio loading, resampling, and validation.

Every function that touches waveforms works with mono float32 numpy arrays
in [-1, 1]-ish amplitude convention (whatever soundfile/librosa decode to),
sampled at an explicit rate passed alongside the array -- never implicit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


class AudioValidationError(ValueError):
    """Raised when an audio file fails the integrity checks in CLAUDE.md's data pipeline."""


def file_checksum(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 checksum of the raw file bytes, used for manifests and cache keys."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Load `path` as mono float32, preserving its native sample rate.

    Returns:
        waveform: shape (num_samples,), dtype float32.
        sample_rate: native sample rate in Hz.
    """
    info = sf.info(str(path))
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    return waveform, int(info.samplerate) if info.samplerate else sample_rate


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Deterministically resample mono float32 `waveform` from `orig_sr` to `target_sr`."""
    if orig_sr == target_sr:
        return np.ascontiguousarray(waveform, dtype=np.float32)
    resampled = librosa.resample(
        waveform.astype(np.float64), orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq"
    )
    return np.ascontiguousarray(resampled, dtype=np.float32)


def validate_waveform(waveform: np.ndarray, *, source_id: str) -> None:
    """Reject NaN, Inf, empty, or all-zero/clipped-flat waveforms (CLAUDE.md data pipeline rule)."""
    if waveform.size == 0:
        raise AudioValidationError(f"{source_id}: waveform is empty")
    if not np.all(np.isfinite(waveform)):
        raise AudioValidationError(f"{source_id}: waveform contains NaN or Inf")
    peak = float(np.max(np.abs(waveform)))
    if peak == 0.0:
        raise AudioValidationError(f"{source_id}: waveform is silent (all zeros)")
    clipped_fraction = float(np.mean(np.abs(waveform) >= 0.999))
    if clipped_fraction > 0.5:
        raise AudioValidationError(
            f"{source_id}: waveform appears clipped ({clipped_fraction:.1%} of samples >= 0.999 abs)"
        )


def load_resample_validate(path: str | Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load, resample to `target_sr`, and validate a file in one deterministic step.

    Returns:
        waveform: mono float32 at `target_sr`.
        orig_sr: the file's native sample rate, for manifest recording.
    """
    waveform, orig_sr = load_mono(path)
    validate_waveform(waveform, source_id=str(path))
    resampled = resample(waveform, orig_sr, target_sr)
    validate_waveform(resampled, source_id=f"{path} (resampled)")
    return resampled, orig_sr
