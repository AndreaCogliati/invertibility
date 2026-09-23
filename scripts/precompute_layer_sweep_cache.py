"""Extracts and caches the 5 candidate MuQ hidden layers (indices 1, 4, 6, 9, 12 -- see
`scripts/inspect_muq_hidden_states.py` and `notes/DECISIONS.md#muq-layer-sweep` for how these
were chosen) for every crop in `data/manifests/more_data_crops.csv` -- the SAME manifest used by
`experiment/muq-to-dac-more-data`, reused unchanged (not rebuilt) per this experiment's explicit
"do not change crop manifests or crop boundaries" requirement.

One MuQ forward pass per crop, `output_hidden_states=True`, extracting all 5 candidate layers at
once (not 5 separate forward passes) -- "preferably in one frozen MuQ forward pass per crop" per
the task spec. DAC latents are NOT recomputed here: they are read directly from the existing,
untouched `data/cache/more_data/muq_dac_cache.pt` (the DAC target definition is explicitly
unchanged by this experiment), keyed by the same crop label.

Writes `data/cache/layer_sweep/muq_layers_cache.pt` (git-ignored, new file -- does NOT overwrite
`data/cache/more_data/muq_dac_cache.pt`). Cache entries are keyed by
`layer_sweep_common.layer_cache_label(...)`, which embeds every field this experiment's spec
requires: source-audio checksum, MuQ checkpoint identifier, MuQ package version, exact
hidden-state index, and dtype (crop boundaries/preprocessing config are embedded in the
crop label itself, via `CropSpec.label`).
"""

from __future__ import annotations

import csv
import importlib.metadata
import os
import tempfile
import time
from pathlib import Path

import torch

from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.gate4_common import CropSpec, load_crop_waveform
from muq_inversion.layer_sweep_common import (
    LAYER_LABELS,
    layer_cache_label,
    select_evenly_spaced_layers,
)
from muq_inversion.logging_utils import DeviceSelectionError, select_device

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/more_data_crops.csv"  # reused unchanged
CACHE_PATH = REPO_ROOT / "data/cache/layer_sweep/muq_layers_cache.pt"
EXISTING_DAC_CACHE_PATH = REPO_ROOT / "data/cache/more_data/muq_dac_cache.pt"  # read-only

SAMPLE_RATE = 24000
MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
NUM_TRANSFORMER_LAYERS = 12  # measured in scripts/inspect_muq_hidden_states.py


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

    if not EXISTING_DAC_CACHE_PATH.exists():
        raise SystemExit(f"{EXISTING_DAC_CACHE_PATH} not found -- this experiment reuses it read-only")
    dac_cache_mtime_before = EXISTING_DAC_CACHE_PATH.stat().st_mtime

    muq_version = importlib.metadata.version("muq")
    muq = FrozenMuQEncoder(MUQ_MODEL_ID, device)

    candidate_indices = select_evenly_spaced_layers(NUM_TRANSFORMER_LAYERS, num_candidates=5)
    print(f"candidate layer indices: {candidate_indices} (labels: {LAYER_LABELS})")

    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"precomputing {len(candidate_indices)} MuQ layers for {len(rows)} crops...")

    cache: dict[str, dict[str, torch.Tensor]] = {}
    dtype_str = "torch.float32"
    start_time = time.perf_counter()
    for i, row in enumerate(rows):
        spec = CropSpec(
            experiment="layer_sweep",
            split=row["split"],
            track_id=row["track_id"],
            genre=row["genre"],
            path=REPO_ROOT / row["path"],
            checksum_sha256=row["checksum_sha256"],
            crop_index=int(row["crop_index"]),
            crop_start_seconds=float(row["crop_start_seconds"]),
            crop_duration_seconds=float(row["crop_duration_seconds"]),
        )
        waveform = load_crop_waveform(spec, SAMPLE_RATE)
        hidden_states = muq.encode_all_hidden_states(waveform)  # one forward pass, all layers

        per_layer: dict[str, torch.Tensor] = {}
        for label, layer_idx in zip(LAYER_LABELS, candidate_indices, strict=True):
            tensor = hidden_states[layer_idx]
            if not torch.isfinite(tensor).all():
                raise SystemExit(f"STOP CONDITION: non-finite MuQ layer {layer_idx} for {spec.label}")
            key = layer_cache_label(spec.label, layer_idx, spec.checksum_sha256, MUQ_MODEL_ID, muq_version, dtype_str)
            per_layer[key] = tensor.detach().clone().cpu()

        cache[spec.label] = per_layer
        if (i + 1) % 200 == 0 or (i + 1) == len(rows):
            elapsed = time.perf_counter() - start_time
            print(f"  {i + 1}/{len(rows)} crops encoded ({elapsed:.1f}s elapsed)")

    elapsed = time.perf_counter() - start_time
    print(f"encoded {len(rows)} crops x {len(candidate_indices)} layers in {elapsed:.2f}s")

    assert EXISTING_DAC_CACHE_PATH.stat().st_mtime == dac_cache_mtime_before, (
        "STOP CONDITION: the existing more_data DAC cache was modified -- this must never happen"
    )

    _save_atomic(
        CACHE_PATH,
        {
            "candidate_layer_indices": candidate_indices,
            "layer_labels": LAYER_LABELS,
            "muq_model_id": MUQ_MODEL_ID,
            "muq_version": muq_version,
            "dtype": dtype_str,
            "crops": cache,
        },
    )
    print(f"wrote cache to {CACHE_PATH}")
    print(f"cache size: {CACHE_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
