"""Pure, testable helpers for `tasks/TASK-14.md`'s full-GTZAN Gate 5 pilot
(`scripts/train_gate5_full_gtzan_wandb.py`), specific to this task and not covered by reused
Gate 3/4 modules:

- `TrainCropSampler`: a deterministic, resumable sampler that draws random-start-offset crops
  broadly across the full training split (not a rotation over a small fixed set, per this task's
  explicit requirement) and tracks per-track exposure counts.
- `paired_delta_stats`: paired correct-vs-control delta statistics (mean/median/proportion
  positive/bootstrap CI) required by this task's "Paired deltas" section.
- `oracle_relative_recovery`: the exact ratio TASK-11/12 used
  (`scripts/plot_gate3_encodec_n32_compute_ablation.py`), pulled out here as a named, tested
  function instead of an inline `100 * v / oracle` expression.

Held-out (validation/test) crop loading and the four conditioning-control constructions are
reused UNCHANGED from `gate4_encodec_common.py` (`load_gate4_manifest`, `load_gate4_examples`,
`build_conditioning`, `CONDITIONING_MODES`) against `data/manifests/gate5_heldout_crops.csv`
(built by `scripts/build_gate5_manifests.py`), which uses the identical CSV schema -- no new
loader or control-construction code is needed for that half of this task.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from muq_inversion.dac_shift_sensitivity_common import cosine_similarity_per_frame
from muq_inversion.data import TrackRecord, read_manifest
from muq_inversion.gate3_encodec_common import find_best_lag_and_align
from muq_inversion.gate4_common import waveform_correlation
from muq_inversion.spectrogram_encoder import LogMelSpectrogramEncoder


@dataclasses.dataclass(frozen=True)
class Gate5Config:
    """Resolved `configs/gate5_full_gtzan.yaml` / `configs/gate5_smoke.yaml` contents. See
    `configs/gate5_full_gtzan.yaml`'s header comment for why this is a purpose-built schema
    rather than the pre-existing `config.py`/`ExperimentConfig` one."""

    task: str
    gate: str
    run_label: str
    seed: int
    device_request: str
    allow_cpu_fallback: bool
    sample_rate: int
    crop_duration_seconds: float
    muq_model_id: str
    encodec_model_id: str
    encodec_bandwidth_kbps: float
    full_gtzan_manifest: str
    heldout_manifest: str
    learning_rate: float
    adam_betas: tuple[float, float]
    cosine_loss_weight: float
    steps: int
    val_every: int
    log_every: int
    early_diagnostic_step: int
    intermediate_checkpoint_steps: tuple[int, ...]
    checkpoint_every: int
    run_dir_name: str
    artifact_dir_name: str
    wandb_group: str
    wandb_tags: tuple[str, ...]
    train_track_limit: int | None = None
    heldout_crop_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["adam_betas"] = list(d["adam_betas"])
        d["intermediate_checkpoint_steps"] = list(d["intermediate_checkpoint_steps"])
        d["wandb_tags"] = list(d["wandb_tags"])
        return d


def load_gate5_config(path: Path) -> Gate5Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = dict(raw)
    raw["adam_betas"] = tuple(raw["adam_betas"])
    raw["intermediate_checkpoint_steps"] = tuple(raw["intermediate_checkpoint_steps"])
    raw["wandb_tags"] = tuple(raw["wandb_tags"])
    known_fields = {f.name for f in dataclasses.fields(Gate5Config)}
    unknown = set(raw) - known_fields
    if unknown:
        raise ValueError(f"{path}: unknown config keys {unknown}")
    return Gate5Config(**raw)


@dataclasses.dataclass(frozen=True)
class TrainCropDraw:
    track_id: str
    genre: str
    relative_path: str
    checksum: str
    crop_start_seconds: float
    crop_duration_seconds: float


@dataclasses.dataclass
class TrainCropSampler:
    """Deterministic, resumable training-crop sampler.

    Per `tasks/TASK-14.md`: "Training must sample from the full training split rather than
    rotate over a small fixed set of clips" and "[use] a deterministic, resumable sampler that
    visits tracks broadly; samples positions across tracks; saves sufficient state for
    continuation; never samples validation or test tracks; reports per-track exposure."

    Track selection is uniform over `tracks` (every training track equally likely per draw --
    "visits tracks broadly"); crop start offset is uniform over the valid range for that track's
    own duration (measured from the manifest's `duration_seconds`, no audio I/O needed to sample)
    -- genuinely different each visit, not a rotation over 1-2 fixed positions per track like
    `more_data_crops.csv`'s train rows. `exposure_counts` accumulates how many times each track
    has been drawn, for the required "concentration on a small subset can be detected" check.
    """

    tracks: list[TrackRecord]
    crop_duration_seconds: float
    rng: random.Random
    exposure_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    num_draws: int = 0

    def __post_init__(self) -> None:
        if not self.tracks:
            raise ValueError("TrainCropSampler needs at least one training track")
        too_short = [
            t.track_id for t in self.tracks if t.duration_seconds < self.crop_duration_seconds
        ]
        if too_short:
            raise ValueError(
                f"STOP CONDITION: {len(too_short)} training track(s) shorter than "
                f"crop_duration_seconds={self.crop_duration_seconds}: {too_short[:5]}"
            )

    def draw(self) -> TrainCropDraw:
        track = self.tracks[self.rng.randrange(len(self.tracks))]
        max_start = track.duration_seconds - self.crop_duration_seconds
        crop_start = self.rng.uniform(0.0, max_start) if max_start > 0 else 0.0

        self.exposure_counts[track.track_id] = self.exposure_counts.get(track.track_id, 0) + 1
        self.num_draws += 1

        return TrainCropDraw(
            track_id=track.track_id,
            genre=track.genre,
            relative_path=track.relative_path,
            checksum=track.checksum,
            crop_start_seconds=crop_start,
            crop_duration_seconds=self.crop_duration_seconds,
        )

    def state_dict(self) -> dict:
        return {
            "rng_state": self.rng.getstate(),
            "exposure_counts": dict(self.exposure_counts),
            "num_draws": self.num_draws,
        }

    def load_state_dict(self, state: dict) -> None:
        self.rng.setstate(state["rng_state"])
        self.exposure_counts = dict(state["exposure_counts"])
        self.num_draws = state["num_draws"]

    def exposure_summary(self) -> dict[str, float]:
        """`tasks/TASK-14.md`'s "reports per-track exposure so concentration on a small subset
        can be detected" requirement: how many distinct tracks have been drawn at least once, and
        the min/max/mean draw count among tracks that have been drawn -- a highly concentrated
        sampler shows a small `tracks_visited` relative to `len(self.tracks)` and/or a large
        max/mean ratio."""
        if not self.exposure_counts:
            return {
                "tracks_visited": 0,
                "tracks_total": len(self.tracks),
                "min_exposure": 0.0,
                "max_exposure": 0.0,
                "mean_exposure": 0.0,
            }
        counts = list(self.exposure_counts.values())
        return {
            "tracks_visited": len(self.exposure_counts),
            "tracks_total": len(self.tracks),
            "min_exposure": float(min(counts)),
            "max_exposure": float(max(counts)),
            "mean_exposure": float(sum(counts) / len(counts)),
        }


def write_limited_heldout_manifest(source_path: Path, dest_path: Path, crop_limit: int) -> None:
    """Writes a bounded subset of `source_path` (the `gate5_heldout_crops.csv` schema) to
    `dest_path`: the first `crop_limit` val rows and first `crop_limit` test rows, in their
    original (deterministic) file order. Used only by the smoke-test config
    (`configs/gate5_smoke.yaml`'s `heldout_crop_limit`) to keep the bounded diagnostic run fast
    without touching `gate4_encodec_common.load_gate4_examples`'s reused, unmodified loading
    logic -- that function always loads every row for a given `experiment` label, so bounding the
    input file itself (rather than the loader) is the smallest change that keeps this diagnostic
    genuinely bounded."""
    import csv

    with open(source_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    val_rows = [r for r in rows if r["split"] == "val"][:crop_limit]
    test_rows = [r for r in rows if r["split"] == "test"][:crop_limit]
    limited = val_rows + test_rows
    if not limited:
        raise ValueError(f"STOP CONDITION: no val/test rows found in {source_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(limited)


def load_train_tracks(manifest_path: Path) -> list[TrackRecord]:
    """Loads only the valid (`not corrupted`), `split == "train"` tracks from the committed
    full-GTZAN manifest (`data/manifests/gtzan_manifest_smoke.csv`) -- the pool
    `TrainCropSampler` draws from. Never includes val/test tracks."""
    records = read_manifest(manifest_path)
    return [r for r in records if r.split == "train" and not r.corrupted]


def load_train_crop_waveform(draw: TrainCropDraw, raw_dir: Path, sample_rate: int) -> torch.Tensor:
    """Loads and resamples `draw`'s track, then crops at `draw.crop_start_seconds` in the
    resampled timeline. Returns (1, crop_samples) float32. Mirrors
    `gate4_encodec_common.load_crop_waveform`'s exact approach for its `Gate4CropSpec`."""
    from muq_inversion.audio import load_resample_validate

    waveform, _ = load_resample_validate(raw_dir / draw.relative_path, sample_rate)
    crop_samples = round(draw.crop_duration_seconds * sample_rate)
    start = round(draw.crop_start_seconds * sample_rate)
    end = start + crop_samples
    if end > len(waveform):
        raise ValueError(
            f"{draw.track_id}: crop [{start},{end}) exceeds resampled length {len(waveform)}"
        )
    return torch.from_numpy(waveform[start:end].copy()).float().unsqueeze(0)


def oracle_relative_recovery(translator_metric: float, oracle_metric: float) -> float:
    """The exact ratio used throughout TASK-11/12
    (`scripts/plot_gate3_encodec_n32_compute_ablation.py`: `100 * v / oracle`), as a fraction in
    [0, 1] rather than a percentage (callers multiply by 100 for display). `oracle_metric` is the
    codec's own encode-decode reconstruction's score against the original (the ceiling); returns
    NaN if the oracle metric is exactly zero (undefined ratio, not a divide-by-zero crash)."""
    if oracle_metric == 0:
        return float("nan")
    return translator_metric / oracle_metric


def compute_held_out_audio_metrics(
    original: torch.Tensor,
    oracle_audio: torch.Tensor,
    predicted_audio: torch.Tensor,
    stft_loss_fn,
    device: torch.device,
) -> dict[str, float]:
    """`tasks/TASK-14.md`'s "Audio" metrics requirement: original-to-oracle, original-to-
    translator, and translator-to-oracle waveform correlation (each pair independently lag-
    aligned via `find_best_lag_and_align`, since a fixed decoder delay would otherwise masquerade
    as reconstruction error -- the same discipline `gate3_encodec_common.compute_oracle_audio_metrics`
    established for the oracle-only case in TASK-11), oracle-relative waveform recovery, log-mel
    similarity and STFT distance (original vs. translator, matching this project's established
    metric definitions), and the lag applied to the original-vs-translator pair.

    All three tensors are 1-D waveforms (already `.squeeze()`d); `oracle_audio` and
    `predicted_audio` need not be pre-aligned to `original` -- this function does that.
    """
    original_cpu, oracle_cpu, predicted_cpu = original.cpu(), oracle_audio.cpu(), predicted_audio.cpu()

    orig_oracle_a, orig_oracle_b, _lag1 = find_best_lag_and_align(original_cpu, oracle_cpu)
    original_to_oracle = waveform_correlation(orig_oracle_b, orig_oracle_a)

    orig_pred_a, orig_pred_b, lag2 = find_best_lag_and_align(original_cpu, predicted_cpu)
    original_to_translator = waveform_correlation(orig_pred_b, orig_pred_a)

    oracle_pred_a, oracle_pred_b, _lag3 = find_best_lag_and_align(oracle_cpu, predicted_cpu)
    translator_to_oracle = waveform_correlation(oracle_pred_b, oracle_pred_a)

    logmel = LogMelSpectrogramEncoder(device)
    mel_a = logmel.encode(orig_pred_a.unsqueeze(0).to(device))
    mel_b = logmel.encode(orig_pred_b.unsqueeze(0).to(device))
    logmel_similarity = cosine_similarity_per_frame(mel_a, mel_b, frame_dim=1)

    with torch.no_grad():
        stft_distance = float(
            stft_loss_fn(orig_pred_b.unsqueeze(0).to(device), orig_pred_a.unsqueeze(0).to(device))[0].item()
        )

    return {
        "original_to_oracle_correlation": original_to_oracle,
        "original_to_translator_correlation": original_to_translator,
        "translator_to_oracle_correlation": translator_to_oracle,
        "oracle_relative_waveform_recovery": oracle_relative_recovery(original_to_translator, original_to_oracle),
        "logmel_similarity": logmel_similarity,
        "stft_distance": stft_distance,
        "lag_applied_original_vs_translator": lag2,
    }


def select_listening_labels(
    per_example: dict[str, dict[str, float]], labels: list[str], metric: str
) -> dict[str, str]:
    """Worst/median/best crop among `labels` by `metric` (ties broken by label, so the choice is
    deterministic). Replaces the original first-N-in-sort-order listening subset, which picked six
    blues crops from three tracks in TASK-14's full run."""
    if not labels:
        raise ValueError("select_listening_labels needs at least one label")
    ranked = sorted(labels, key=lambda lbl: (per_example[lbl][metric], lbl))
    return {"worst": ranked[0], "median": ranked[len(ranked) // 2], "best": ranked[-1]}


def write_listening_artifacts(
    labels: list[str],
    heldout_examples: dict,
    model: torch.nn.Module,
    encodec,
    frame_ratio: int,
    seed: int,
    out_dir: Path,
    sample_rate: int,
) -> None:
    """Writes original, EnCodec oracle, and all four conditioning reconstructions for each label.
    Shared by the training script and `scripts/export_gate5_listening.py` so both produce the same
    files; the control RNG is seeded once and consumed in `labels` order, and `shuffled_across`
    draws from the full held-out pool in both."""
    import soundfile as sf

    from muq_inversion.gate4_encodec_common import (
        CONDITIONING_MODES,
        build_conditioning,
    )

    def _write(path: Path, waveform: torch.Tensor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), waveform.detach().cpu().numpy().astype(np.float32), sample_rate)

    rng = random.Random(seed)
    pool = list(heldout_examples.values())
    model.eval()
    for label in labels:
        ex = heldout_examples[label]
        with torch.no_grad():
            oracle_audio = encodec.decode(ex.target.transpose(1, 2)).squeeze(0)
        _write(out_dir / f"{label}_original.wav", ex.waveform.squeeze(0))
        _write(out_dir / f"{label}_encodec_oracle.wav", oracle_audio)
        for mode in CONDITIONING_MODES:
            with torch.no_grad():
                conditioned = build_conditioning(mode, ex, pool, frame_ratio, rng)
                pred_audio = encodec.decode(model(conditioned).transpose(1, 2)).squeeze(0)
            _write(out_dir / f"{label}_{mode}.wav", pred_audio)


def paired_delta_stats(
    correct_values: list[float], control_values: list[float], seed: int = 0, num_bootstrap: int = 2000
) -> dict[str, float]:
    """`tasks/TASK-14.md`'s "Paired deltas" requirement: for matched (correct, control) pairs
    over the same held-out examples, `correct - control`, then mean/median/proportion-positive
    and a deterministic bootstrap 95% confidence interval (seeded `np.random.default_rng`, so
    this is reproducible across runs, not a source of nondeterminism in the report). Requires
    `len(correct_values) == len(control_values)` (paired by construction, same example order)."""
    if len(correct_values) != len(control_values):
        raise ValueError(
            f"paired_delta_stats requires matched pairs: {len(correct_values)} correct values, "
            f"{len(control_values)} control values"
        )
    if not correct_values:
        raise ValueError("paired_delta_stats requires at least one paired example")

    deltas = np.array(correct_values) - np.array(control_values)
    rng = np.random.default_rng(seed)
    n = len(deltas)
    bootstrap_means = np.empty(num_bootstrap)
    for i in range(num_bootstrap):
        sample = deltas[rng.integers(0, n, size=n)]
        bootstrap_means[i] = sample.mean()
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

    return {
        "n": n,
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "proportion_positive": float((deltas > 0).mean()),
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "bootstrap_seed": seed,
        "num_bootstrap": num_bootstrap,
    }
