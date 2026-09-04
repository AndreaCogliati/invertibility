"""CLI entry points. Only `doctor` (Gate 0) and `prepare-data` (Gate 1) are
implemented so far -- extract/train/evaluate belong to later gates."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, config_to_dict, load_config
from .data import (
    assign_splits,
    download_and_extract,
    find_duplicate_checksums,
    index_gtzan,
    verify_resample_sample,
    write_manifest,
)
from .logging_utils import (
    DeviceSelectionError,
    capture_environment_manifest,
    select_device,
)
from .models.generator import MiniGenerator

REPO_ROOT = Path(__file__).resolve().parents[2]


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cmd_doctor(config: ExperimentConfig) -> int:
    print("=== Gate 0: Environment Doctor ===")
    env_manifest = capture_environment_manifest(REPO_ROOT)
    for key, value in env_manifest.items():
        print(f"{key}: {value}")

    print(f"machine architecture: {platform.machine()}")

    try:
        device = select_device(config.environment.device, config.environment.allow_cpu_fallback)
    except DeviceSelectionError as exc:
        print(f"DEVICE SELECTION FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"selected device: {device}")

    _set_seeds(config.environment.seed)

    print("\n--- MuQ encoder smoke test ---")
    from .encoder import FrozenMuQEncoder

    encoder = FrozenMuQEncoder(config.muq.model_id, device)
    duration_seconds = 2.0
    num_samples = int(duration_seconds * config.muq.sample_rate)
    torch.manual_seed(config.environment.seed)
    synthetic_waveform = torch.randn(1, num_samples, dtype=torch.float32)

    probe = encoder.probe(synthetic_waveform, config.muq.sample_rate)
    for field_name, value in probe.__dict__.items():
        print(f"muq.{field_name}: {value}")

    if not probe.is_finite:
        print("MuQ produced non-finite output on a synthetic clip. Stopping.", file=sys.stderr)
        return 1

    # Determinism check: same input, same output.
    probe_repeat = encoder.probe(synthetic_waveform, config.muq.sample_rate)
    same_shape = probe.feature_shape == probe_repeat.feature_shape
    print(f"muq.deterministic_shape_repeat: {same_shape}")

    print("\n--- Decoder shape-only forward pass ---")
    generator = MiniGenerator(
        input_channels=probe.feature_channels,
        base_channels=config.decoder.base_channels,
        upsample_rates=config.decoder.upsample_rates,
    ).to(device)
    generator.eval()

    dummy_features = torch.randn(1, probe.num_frames, probe.feature_channels, device=device)
    with torch.no_grad():
        waveform_out = generator(dummy_features)
    expected_samples = probe.num_frames * generator.total_upsample
    print(f"decoder.input_shape: {tuple(dummy_features.shape)}")
    print(f"decoder.output_shape: {tuple(waveform_out.shape)}")
    print(f"decoder.total_upsample (provisional): {generator.total_upsample}")
    print(f"decoder.expected_output_samples: {expected_samples}")

    if waveform_out.shape[-1] != expected_samples:
        print("Decoder output length does not match its own upsample product.", file=sys.stderr)
        return 1
    if not torch.isfinite(waveform_out).all():
        print("Decoder produced non-finite output on a shape-only pass.", file=sys.stderr)
        return 1

    print("\nGate 0 PASSED.")
    return 0


def cmd_prepare_data(config: ExperimentConfig) -> int:
    print("=== Gate 1: Data Integrity ===")
    raw_dir = REPO_ROOT / config.data.raw_dir
    manifests_dir = REPO_ROOT / config.data.manifests_dir

    print(f"raw_dir: {raw_dir}")
    print(f"source: {config.data.source}")
    download_and_extract(raw_dir)

    print("\n--- Indexing ---")
    records = index_gtzan(raw_dir, config.data.genres)
    print(f"indexed {len(records)} files")
    if len(records) != config.data.expected_track_count:
        print(
            f"WARNING: expected {config.data.expected_track_count} tracks, found {len(records)}",
            file=sys.stderr,
        )

    corrupted = [r for r in records if r.corrupted]
    print(f"known-corrupted files flagged (excluded from splits/usage): {[r.track_id for r in corrupted]}")

    duplicates = find_duplicate_checksums(records)
    if duplicates:
        print(f"WARNING: duplicate checksums found: {duplicates}", file=sys.stderr)
    else:
        print("no duplicate checksums found")

    print("\n--- Deterministic split ---")
    records = assign_splits(
        records,
        config.data.train_fraction,
        config.data.val_fraction,
        config.data.test_fraction,
        config.data.split_seed,
    )
    split_counts: dict[str, int] = {}
    for r in records:
        split_counts[r.split] = split_counts.get(r.split, 0) + 1
    print(f"split counts: {split_counts}")

    manifest_path = manifests_dir / f"gtzan_manifest_{config.name}.csv"
    if manifest_path.exists():
        print(
            f"Manifest already exists at {manifest_path}; refusing to overwrite "
            "(CLAUDE.md rule 8: never overwrite manifests from a previous run).",
            file=sys.stderr,
        )
        return 1
    write_manifest(records, manifest_path)
    print(f"manifest written: {manifest_path}")

    print("\n--- Resample verification sample ---")
    sample_size = min(20, len(records))
    verifications = verify_resample_sample(
        records, raw_dir, config.data, sample_size, config.data.split_seed
    )
    for v in verifications:
        print(
            f"{v.track_id}: {v.native_sample_rate}Hz/{v.native_num_samples} -> "
            f"{v.target_sample_rate}Hz/{v.resampled_num_samples}, "
            f"finite={v.is_finite}, peak={v.peak_amplitude:.4f}"
        )
    if not all(v.is_finite for v in verifications):
        print("Non-finite samples found during resample verification.", file=sys.stderr)
        return 1

    manifest_summary = {
        "total_records": len(records),
        "corrupted": len(corrupted),
        "duplicates": len(duplicates),
        "split_counts": split_counts,
        "resample_verified": len(verifications),
    }
    print("\nGate 1 summary:")
    print(json.dumps(manifest_summary, indent=2))
    print("\nGate 1 PASSED.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="muq_inversion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Gate 0: environment doctor")
    doctor_parser.add_argument("--config", required=True, type=Path)

    prepare_parser = subparsers.add_parser("prepare-data", help="Gate 1: data integrity")
    prepare_parser.add_argument("--config", required=True, type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    print(f"config: {args.config} ({config.name})")
    print(json.dumps(config_to_dict(config), indent=2, default=str))
    print()

    if args.command == "doctor":
        return cmd_doctor(config)
    if args.command == "prepare-data":
        return cmd_prepare_data(config)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
