"""Pure, testable helpers for `tasks/TASK-13.md`'s Gate 4 revalidation on the merged (post-
`experiment/muq-layer-sweep`) architecture (`scripts/train_gate4_encodec_wandb.py`).

Reuses the exact merged pipeline validated in TASK-11/TASK-12 unmodified: `FrozenMuQEncoder`,
`FrozenEncodecEncoder`, `SpectrogramToDacTranslator` (generic name; here instantiated with MuQ's
1024 channels as input and EnCodec's 128 channels as output), `evaluate_prediction`/
`aggregate_metrics_across_examples`/`rank_examples_by_metric` from `gate3_spectrogram_common.py`
(duck-typed on the EnCodec encoder exactly as TASK-11 established), and `find_best_lag_and_align`
from `gate3_encodec_common.py`. This module only adds the Gate-4-specific data loading (the
`gate4_crops.csv` manifest, MuQ input instead of log-mel) and the four conditioning-control
constructions CLAUDE.md's Evaluation and Controls section and `tasks/TASK-13.md` require:

    "correct":         the clip's own MuQ frames, in order.
    "shuffled_across":  a different clip's MuQ frames entirely.
    "shuffled_time":    the clip's own MuQ frames, permuted in time.
    "constant_zero":    an all-zero conditioning tensor.

All four are constructed at MuQ's native (pre-repeat-interleave) frame rate, then upsampled to
EnCodec's frame rate via the same `repeat_interleave` alignment strategy every MuQ-to-codec
experiment in this project has used since Gate 3 -- the conditioning intervention is the only
thing that changes between conditions, per this task's explicit requirement.
"""

from __future__ import annotations

import csv
import dataclasses
import random
from pathlib import Path

import torch

from muq_inversion.audio import load_resample_validate
from muq_inversion.encodec_encoder import FrozenEncodecEncoder
from muq_inversion.encoder import FrozenMuQEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
CONDITIONING_MODES = ("correct", "shuffled_across", "shuffled_time", "constant_zero")


@dataclasses.dataclass(frozen=True)
class Gate4CropSpec:
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
        return f"{self.split}_{self.track_id}_c{self.crop_index}"


@dataclasses.dataclass(frozen=True)
class Gate4Example:
    spec: Gate4CropSpec
    muq: torch.Tensor  # (1, num_muq_frames, 1024) -- MuQ's native 25Hz timeline
    target: torch.Tensor  # (1, num_encodec_frames, 128) -- EnCodec quantized latent, frames-first
    waveform: torch.Tensor  # (1, num_samples) -- the raw crop the target was extracted from


def load_gate4_manifest(manifest_path: Path, experiment: str) -> list[Gate4CropSpec]:
    specs: list[Gate4CropSpec] = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["experiment"] != experiment:
                continue
            specs.append(
                Gate4CropSpec(
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


def load_crop_waveform(spec: Gate4CropSpec, sample_rate: int) -> torch.Tensor:
    """Loads and resamples the full track, then crops at `spec.crop_start_seconds` in the
    resampled timeline. Returns (1, crop_samples) float32."""
    waveform, _ = load_resample_validate(spec.path, sample_rate)
    crop_samples = round(spec.crop_duration_seconds * sample_rate)
    start = round(spec.crop_start_seconds * sample_rate)
    end = start + crop_samples
    if end > len(waveform):
        raise ValueError(
            f"{spec.track_id}: crop [{start},{end}) exceeds resampled length {len(waveform)}"
        )
    crop = waveform[start:end]
    return torch.from_numpy(crop.copy()).float().unsqueeze(0)


def load_gate4_examples(
    manifest_path: Path,
    experiment: str,
    muq: FrozenMuQEncoder,
    encodec: FrozenEncodecEncoder,
    sample_rate: int,
    device: torch.device,
) -> tuple[dict[str, Gate4Example], int]:
    """Precomputes MuQ (input) and EnCodec (target) latents for every crop in `experiment`.
    Returns the examples keyed by `Gate4CropSpec.label` and the verified integer frame ratio
    (EnCodec frames per MuQ frame). Both encoders' outputs are `.clone()`-d before returning:
    MuQ's `encode()` is not `inference_mode`-decorated (see `encoder.py`), but EnCodec's is, and
    its raw output would otherwise be an inference tensor unusable as a later autograd target
    (the exact bug TASK-11 diagnosed and fixed in `gate3_encodec_common.py`)."""
    specs = load_gate4_manifest(manifest_path, experiment)
    if not specs:
        raise ValueError(f"no crops found for experiment {experiment!r} in {manifest_path}")

    examples: dict[str, Gate4Example] = {}
    frame_ratio: int | None = None
    for spec in specs:
        waveform = load_crop_waveform(spec, sample_rate).to(device)
        muq_feat = muq.encode(waveform).detach().clone()  # (1, T_muq, 1024)
        target = encodec.encode(waveform).transpose(1, 2).clone()  # (1,128,T)->(1,T,128)

        ratio = target.shape[1] / muq_feat.shape[1]
        if ratio != round(ratio):
            raise ValueError(
                f"STOP CONDITION: non-integer MuQ->EnCodec frame ratio for {spec.label}: {ratio}"
            )
        ratio = round(ratio)
        if frame_ratio is None:
            frame_ratio = ratio
        elif ratio != frame_ratio:
            raise ValueError(
                f"STOP CONDITION: frame ratio changed across crops: {ratio} != {frame_ratio}"
            )
        if not torch.isfinite(muq_feat).all() or not torch.isfinite(target).all():
            raise ValueError(f"STOP CONDITION: non-finite MuQ or EnCodec latent for {spec.label}")

        examples[spec.label] = Gate4Example(spec=spec, muq=muq_feat, target=target, waveform=waveform)
    assert frame_ratio is not None
    return examples, frame_ratio


def build_conditioning(
    mode: str,
    example: Gate4Example,
    other_examples: list[Gate4Example],
    frame_ratio: int,
    rng: random.Random,
) -> torch.Tensor:
    """Computes the repeat-interleaved (EnCodec-rate) conditioning tensor for one example under
    `mode`. The permutation/selection itself always happens at MuQ's native frame rate, then the
    result is upsampled by `frame_ratio` -- identical alignment strategy to the "correct" case, so
    only the conditioning content differs between modes, per this task's explicit requirement.
    """
    if mode == "correct":
        muq_feat = example.muq
    elif mode == "shuffled_across":
        candidates = [e for e in other_examples if e.spec.track_id != example.spec.track_id]
        if not candidates:
            raise ValueError(f"no other-track candidates available for {example.spec.label}")
        source = candidates[rng.randrange(len(candidates))]
        muq_feat = source.muq
        if muq_feat.shape[1] != example.muq.shape[1]:
            # Align by truncating to the shorter length, then re-checking non-emptiness -- crops
            # in this manifest all share crop_duration_seconds, so this should not trigger; kept
            # as a defensive, documented fallback rather than a silent shape-mismatch crash.
            n = min(muq_feat.shape[1], example.muq.shape[1])
            if n == 0:
                raise ValueError(f"STOP CONDITION: zero-length shuffled_across source for {example.spec.label}")
            muq_feat = muq_feat[:, :n, :]
    elif mode == "shuffled_time":
        perm = torch.randperm(example.muq.shape[1], generator=torch.Generator().manual_seed(rng.randrange(2**31)))
        muq_feat = example.muq[:, perm, :]
    elif mode == "constant_zero":
        muq_feat = torch.zeros_like(example.muq)
    else:
        raise ValueError(f"unknown conditioning mode: {mode!r}")

    return muq_feat.repeat_interleave(frame_ratio, dim=1)
