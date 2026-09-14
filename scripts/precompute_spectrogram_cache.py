"""Precomputes and caches log-mel spectrogram features for every crop in
`data/manifests/more_data_crops.csv` (the exact manifest/split used by
`experiment/muq-to-dac-more-data`, the current best MuQ-to-DAC model -- see `tasks/TASK-01.md`).

DAC latents are NOT recomputed here: `data/cache/more_data/muq_dac_cache.pt` already holds them
for this exact manifest and is reused read-only, unmodified, by the training script -- the same
"reuse the DAC cache, only swap the input representation" pattern as
`scripts/precompute_layer_sweep_cache.py`.

Writes one consolidated file, `data/cache/spectrogram_control/mel_cache.pt` (git-ignored),
containing a dict keyed by crop label -> log-mel tensor (1, 150, 128), already truncated to
DAC's exact frame count. Written atomically, per this project's standing cache discipline.
"""

from __future__ import annotations

import csv
import os
import tempfile
import time
from pathlib import Path

import torch

from muq_inversion.gate4_common import CropSpec, load_crop_waveform
from muq_inversion.logging_utils import DeviceSelectionError, select_device
from muq_inversion.spectrogram_encoder import (
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    LogMelSpectrogramEncoder,
    align_to_dac_frames,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/more_data_crops.csv"
DAC_CACHE_PATH = REPO_ROOT / "data/cache/more_data/muq_dac_cache.pt"  # reused, read-only
CACHE_PATH = REPO_ROOT / "data/cache/spectrogram_control/mel_cache.pt"

SAMPLE_RATE = 24000


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

    if not DAC_CACHE_PATH.exists():
        raise SystemExit(f"{DAC_CACHE_PATH} not found -- this experiment reuses it from experiment/muq-to-dac-more-data")
    dac_cache = torch.load(DAC_CACHE_PATH, map_location=device, weights_only=False)
    dac_frame_counts = {label: v["dac"].shape[1] for label, v in dac_cache["crops"].items()}

    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"precomputing log-mel spectrograms for {len(rows)} crops "
          f"(n_fft={N_FFT}, hop_length={HOP_LENGTH}, n_mels={N_MELS})...")

    mel_encoder = LogMelSpectrogramEncoder(device)

    cache: dict[str, torch.Tensor] = {}
    seen_labels: set[str] = set()
    start_time = time.perf_counter()
    for i, row in enumerate(rows):
        spec = CropSpec(
            experiment="spectrogram_control",
            split=row["split"],
            track_id=row["track_id"],
            genre=row["genre"],
            path=REPO_ROOT / row["path"],
            checksum_sha256=row["checksum_sha256"],
            crop_index=int(row["crop_index"]),
            crop_start_seconds=float(row["crop_start_seconds"]),
            crop_duration_seconds=float(row["crop_duration_seconds"]),
        )
        label = spec.label
        assert label not in seen_labels, f"STOP CONDITION: duplicate crop label {label}"
        seen_labels.add(label)
        assert label in dac_frame_counts, f"STOP CONDITION: {label} missing from DAC cache -- manifest mismatch"

        waveform = load_crop_waveform(spec, SAMPLE_RATE)
        mel = mel_encoder.encode(waveform)
        mel = align_to_dac_frames(mel, dac_frame_counts[label])

        if not torch.isfinite(mel).all():
            raise SystemExit(f"STOP CONDITION: non-finite spectrogram for {label}")

        cache[label] = mel.detach().clone().cpu()
        if (i + 1) % 200 == 0 or (i + 1) == len(rows):
            elapsed = time.perf_counter() - start_time
            print(f"  {i + 1}/{len(rows)} crops encoded ({elapsed:.1f}s elapsed)")

    elapsed = time.perf_counter() - start_time
    print(f"encoded {len(rows)} crops in {elapsed:.2f}s")

    _save_atomic(
        CACHE_PATH,
        {
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "crops": cache,
        },
    )
    print(f"wrote cache to {CACHE_PATH}")


if __name__ == "__main__":
    main()
