"""Precomputes and caches MuQ features and DAC latents for every crop in
`data/manifests/more_data_crops.csv` (2000 crops, ~792/94/113 train/val/test tracks -- nearly the
full GTZAN dataset), so the training script never re-runs either frozen encoder during training.

Writes one consolidated file, `data/cache/more_data/muq_dac_cache.pt` (git-ignored), containing a
dict keyed by crop label -> `{"muq": tensor, "dac": tensor}`. Written atomically, matching this
project's standing checkpoint/cache discipline.

Near-identical to `experiment/muq-to-dac-scale-data`'s `precompute_scale_data_cache.py`; only the
manifest/cache paths and experiment label differ.
"""

from __future__ import annotations

import csv
import os
import tempfile
import time
from pathlib import Path

import torch

from muq_inversion.dac_encoder import FrozenDacEncoder
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.gate4_common import CropSpec, load_crop_waveform
from muq_inversion.logging_utils import DeviceSelectionError, select_device

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/more_data_crops.csv"
CACHE_PATH = REPO_ROOT / "data/cache/more_data/muq_dac_cache.pt"

SAMPLE_RATE = 24000
MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
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

    muq = FrozenMuQEncoder(MUQ_MODEL_ID, device)
    dac = FrozenDacEncoder(DAC_MODEL_ID, device)

    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"precomputing MuQ + DAC for {len(rows)} crops...")
    cache: dict[str, dict[str, torch.Tensor]] = {}
    frame_ratio: int | None = None
    seen_labels: set[str] = set()
    start_time = time.perf_counter()
    for i, row in enumerate(rows):
        spec = CropSpec(
            experiment="more_data",
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

        waveform = load_crop_waveform(spec, SAMPLE_RATE)
        muq_feat = muq.encode(waveform)
        dac_feat = dac.encode(waveform)

        if not torch.isfinite(muq_feat).all() or not torch.isfinite(dac_feat).all():
            raise SystemExit(f"STOP CONDITION: non-finite latents for {label}")

        ratio = dac_feat.shape[-1] / muq_feat.shape[1]
        assert ratio == round(ratio), f"STOP CONDITION: non-integer frame ratio for {label}: {ratio}"
        ratio = round(ratio)
        if frame_ratio is None:
            frame_ratio = ratio
        assert ratio == frame_ratio, f"STOP CONDITION: frame ratio changed for {label}: {ratio} != {frame_ratio}"

        cache[label] = {
            "muq": muq_feat.detach().clone().cpu(),
            "dac": dac_feat.transpose(1, 2).detach().clone().cpu(),
        }
        if (i + 1) % 200 == 0 or (i + 1) == len(rows):
            elapsed = time.perf_counter() - start_time
            print(f"  {i + 1}/{len(rows)} crops encoded ({elapsed:.1f}s elapsed)")

    elapsed = time.perf_counter() - start_time
    print(f"encoded {len(rows)} crops in {elapsed:.2f}s, frame_ratio={frame_ratio}")

    _save_atomic(CACHE_PATH, {"frame_ratio": frame_ratio, "crops": cache})
    print(f"wrote cache to {CACHE_PATH}")


if __name__ == "__main__":
    main()
