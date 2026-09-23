"""Shared data structures and evaluation logic for the Gate 4 generalization experiment
(`experiment/muq-to-dac-minimal-gate4`).

Reuses the exact model (`TemporalConvNet`), encoders (`FrozenMuQEncoder`, `FrozenDacEncoder`),
and temporal-alignment strategy (repeat each MuQ frame to match DAC's frame rate) validated in
Gate 3 (`experiment/muq-to-dac-temporal-conv`) without modification. This module only adds the
data-splitting, control-condition, and metric machinery needed to test generalization -- no
architecture, loss, or optimizer change.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import numpy as np
import torch

from muq_inversion.audio import load_resample_validate

REPO_ROOT = Path(__file__).resolve().parents[1].parent


@dataclasses.dataclass(frozen=True)
class CropSpec:
    experiment: str
    split: str
    track_id: str
    genre: str
    path: Path
    checksum_sha256: str
    crop_index: int
    crop_start_seconds: float
    crop_duration_seconds: float

    @property
    def label(self) -> str:
        """Descriptive identifier for filenames: split, track, crop start -- per this
        experiment's "descriptive filenames" requirement."""
        return f"{self.split}_{self.track_id}_t{self.crop_start_seconds:.1f}s"


def load_gate4_manifest(manifest_path: Path, experiment: str) -> list[CropSpec]:
    specs: list[CropSpec] = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["experiment"] != experiment:
                continue
            specs.append(
                CropSpec(
                    experiment=row["experiment"],
                    split=row["split"],
                    track_id=row["track_id"],
                    genre=row["genre"],
                    path=REPO_ROOT / row["path"],
                    checksum_sha256=row["checksum_sha256"],
                    crop_index=int(row["crop_index"]),
                    crop_start_seconds=float(row["crop_start_seconds"]),
                    crop_duration_seconds=float(row["crop_duration_seconds"]),
                )
            )
    return specs


def load_crop_waveform(spec: CropSpec, sample_rate: int) -> torch.Tensor:
    """Loads and resamples the full track, then crops at `spec.crop_start_seconds` in the
    resampled timeline (resampling preserves time alignment, so `start_seconds * sample_rate`
    indexes correctly). Returns (1, crop_samples) float32."""
    waveform, _ = load_resample_validate(spec.path, sample_rate)
    crop_samples = round(spec.crop_duration_seconds * sample_rate)
    start = round(spec.crop_start_seconds * sample_rate)
    end = start + crop_samples
    if end > len(waveform):
        raise ValueError(
            f"{spec.track_id}: crop [{start},{end}) exceeds resampled length {len(waveform)}"
        )
    crop = waveform[start:end]
    return torch.from_numpy(crop).float().unsqueeze(0)


def shifted_time_control(muq_latent: torch.Tensor, shift_frames: int) -> torch.Tensor:
    """Circular shift along the time axis by a fixed nonzero offset -- control #3
    ("MuQ features shifted in time by a fixed nonzero offset")."""
    return torch.roll(muq_latent, shifts=shift_frames, dims=1)


def waveform_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    n = min(len(a_np), len(b_np))
    if n < 2:
        return float("nan")
    return float(np.corrcoef(a_np[:n], b_np[:n])[0, 1])
