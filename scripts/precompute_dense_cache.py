"""Precomputes and caches DAC latents + log-mel spectrograms for every crop in
`data/manifests/dense_crops.csv` (14976 crops, ~100% GTZAN coverage -- `tasks/TASK-02.md`).

Unlike `precompute_more_data_cache.py` / `precompute_spectrogram_cache.py` (which call
`load_crop_waveform` once per crop, reloading and resampling the *entire track* from disk each
time), this script resamples each of the 999 tracks exactly once and slices all ~15 of its crops
from the resulting in-memory waveform. With ~15 crops/track here (vs. 2 in the prior scripts),
per-crop reloading would mean ~15x redundant disk reads and resampling -- worth avoiding at this
scale, per CLAUDE.md's resource-discipline guidance, without changing any experimental variable
(the resulting per-crop audio is byte-identical either way, since resampling is deterministic).

Both representations use their TASK-01-unchanged parameters (`spectrogram_encoder.py`'s hop=320/
n_mels=128/etc., `dac_encoder.py`'s `descript/dac_24khz`). Writes one consolidated file,
`data/cache/dense/mel_dac_cache.pt` (git-ignored, ~10.35GB, estimated in
`build_dense_gtzan_manifest.py`'s docstring), keyed by crop label -> {"dac": tensor, "mel":
tensor}. Written atomically per this project's standing cache discipline.
"""

from __future__ import annotations

import csv
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch

from muq_inversion.audio import load_resample_validate
from muq_inversion.dac_encoder import FrozenDacEncoder
from muq_inversion.gate4_common import CropSpec
from muq_inversion.logging_utils import DeviceSelectionError, select_device
from muq_inversion.spectrogram_encoder import (
    LogMelSpectrogramEncoder,
    align_to_dac_frames,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/dense_crops.csv"
CACHE_PATH = REPO_ROOT / "data/cache/dense/mel_dac_cache.pt"

SAMPLE_RATE = 24000
DAC_MODEL_ID = "descript/dac_24khz"


def _save_atomic(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def main() -> None:
    try:
        device = select_device(requested="auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")

    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"precomputing DAC + log-mel for {len(rows)} crops across "
          f"{len({r['track_id'] for r in rows})} tracks (one resample per track)...")

    by_track: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[row["track_id"]].append(row)

    dac = FrozenDacEncoder(DAC_MODEL_ID, device)
    mel_encoder = LogMelSpectrogramEncoder(device)

    cache: dict[str, dict[str, torch.Tensor]] = {}
    seen_labels: set[str] = set()
    start_time = time.perf_counter()
    for tracks_done, (track_id, track_rows) in enumerate(by_track.items(), start=1):
        first = track_rows[0]
        path = REPO_ROOT / first["path"]
        waveform_np, _ = load_resample_validate(path, SAMPLE_RATE)
        waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)  # (1, num_samples)

        for row in track_rows:
            spec = CropSpec(
                experiment="dense",
                split=row["split"],
                track_id=track_id,
                genre=row["genre"],
                path=REPO_ROOT / row["path"],
                checksum_sha256=row["checksum_sha256"],
                crop_index=int(row["crop_index"]),
                crop_start_seconds=float(row["crop_start_seconds"]),
                crop_duration_seconds=float(row["crop_duration_seconds"]),
            )
            start = round(spec.crop_start_seconds * SAMPLE_RATE)
            crop_samples = round(spec.crop_duration_seconds * SAMPLE_RATE)
            end = start + crop_samples
            if end > waveform.shape[-1]:
                raise SystemExit(
                    f"STOP CONDITION: {track_id} crop [{start},{end}) exceeds resampled length {waveform.shape[-1]}"
                )
            crop_wave = waveform[:, start:end]

            label = spec.label
            assert label not in seen_labels, f"STOP CONDITION: duplicate crop label {label}"
            seen_labels.add(label)

            dac_feat = dac.encode(crop_wave).transpose(1, 2)  # (1, num_frames, 1024)
            mel_feat = mel_encoder.encode(crop_wave)  # (1, num_frames+1, 128) before alignment
            mel_feat = align_to_dac_frames(mel_feat, dac_feat.shape[1])

            if not torch.isfinite(dac_feat).all() or not torch.isfinite(mel_feat).all():
                raise SystemExit(f"STOP CONDITION: non-finite latent for {label}")

            cache[label] = {
                "dac": dac_feat.detach().clone().cpu(),
                "mel": mel_feat.detach().clone().cpu(),
            }

        if tracks_done % 100 == 0 or tracks_done == len(by_track):
            elapsed = time.perf_counter() - start_time
            print(f"  {tracks_done}/{len(by_track)} tracks ({len(cache)}/{len(rows)} crops) "
                  f"encoded ({elapsed:.1f}s elapsed)")

    elapsed = time.perf_counter() - start_time
    print(f"encoded {len(cache)} crops across {len(by_track)} tracks in {elapsed:.2f}s")

    _save_atomic(CACHE_PATH, {"crops": cache})
    print(f"wrote cache to {CACHE_PATH} ({CACHE_PATH.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
