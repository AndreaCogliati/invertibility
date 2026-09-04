"""CLI entry points. `doctor` (Gate 0), `prepare-data` (Gate 1), and `extract`
(Gate 2) are implemented so far -- train/evaluate belong to later gates."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch

from .audio import load_resample_validate
from .cache import CacheKeyInputs, cache_path, compute_cache_key, save_atomic
from .cache import load as load_cache
from .config import ExperimentConfig, config_to_dict, load_config
from .data import (
    assign_splits,
    deterministic_unit_interval,
    download_and_extract,
    find_duplicate_checksums,
    index_gtzan,
    read_manifest,
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


def cmd_extract(config: ExperimentConfig) -> int:
    print("=== Gate 2: Feature Contract ===")
    manifest_path = REPO_ROOT / config.data.manifests_dir / f"gtzan_manifest_{config.name}.csv"
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}. Run prepare-data first.", file=sys.stderr)
        return 1
    records = read_manifest(manifest_path)
    raw_dir = REPO_ROOT / config.data.raw_dir
    cache_dir = REPO_ROOT / config.cache.cache_dir
    fc_config = config.feature_contract

    try:
        device = select_device(config.environment.device, config.environment.allow_cpu_fallback)
    except DeviceSelectionError as exc:
        print(f"DEVICE SELECTION FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"selected device: {device}")

    _set_seeds(config.environment.seed)

    from .encoder import FrozenMuQEncoder, measure_feature_geometry

    encoder = FrozenMuQEncoder(config.muq.model_id, device)

    usable = [r for r in records if not r.corrupted]
    ordered = sorted(
        usable, key=lambda r: deterministic_unit_interval(r.track_id, config.data.split_seed)
    )
    probe_tracks = ordered[: fc_config.probe_track_count]
    checksum_by_track_id = {r.track_id: r.checksum for r in probe_tracks}
    print(f"probing {len(probe_tracks)} tracks: {[t.track_id for t in probe_tracks]}")

    print("\n--- Extracting across several durations ---")
    probes: list[tuple[str, str, object, torch.Tensor]] = []
    for record in probe_tracks:
        path = raw_dir / record.relative_path
        waveform_np, _orig_sr = load_resample_validate(path, config.data.target_sample_rate)
        full_duration = len(waveform_np) / config.data.target_sample_rate

        durations = sorted(
            {d for d in fc_config.probe_durations_seconds if d <= full_duration} | {full_duration}
        )
        for duration in durations:
            is_full = duration == full_duration
            num_samples = (
                len(waveform_np) if is_full else round(duration * config.data.target_sample_rate)
            )
            cropped = waveform_np[:num_samples]
            waveform_tensor = torch.from_numpy(cropped).unsqueeze(0)
            features = encoder.encode(waveform_tensor)
            result = measure_feature_geometry(features, cropped.shape[0], config.data.target_sample_rate)
            label = "full" if is_full else f"{duration:g}s"
            probes.append((record.track_id, label, result, features))
            print(
                f"{record.track_id} [{label}]: samples={result.input_num_samples} "
                f"frames={result.num_frames} fps={result.frames_per_second:.4f} "
                f"samples/frame={result.samples_per_frame:.4f} channels={result.feature_channels} "
                f"finite={result.is_finite}"
            )
            if not result.is_finite:
                print("Non-finite MuQ output during feature-contract probing.", file=sys.stderr)
                return 1

    print("\n--- Frame-rate consistency ---")
    fps_values = [r.frames_per_second for _, _, r, _ in probes]
    channel_values = {r.feature_channels for _, _, r, _ in probes}
    fps_min, fps_max = min(fps_values), max(fps_values)
    print(f"frames_per_second range: [{fps_min:.4f}, {fps_max:.4f}]")
    if len(channel_values) != 1:
        print(f"Inconsistent feature_channels across probes: {channel_values}", file=sys.stderr)
        return 1
    if fps_max - fps_min > fc_config.frame_rate_tolerance_hz:
        print(
            f"Frame rate inconsistent across durations/tracks: spread {fps_max - fps_min:.4f} Hz "
            f"exceeds tolerance {fc_config.frame_rate_tolerance_hz} Hz",
            file=sys.stderr,
        )
        return 1

    mean_samples_per_frame = sum(r.samples_per_frame for _, _, r, _ in probes) / len(probes)
    print(f"mean samples_per_frame: {mean_samples_per_frame:.4f}")

    print("\n--- Alignment check ---")
    for track_id, label, result, _ in probes:
        expected_frames = round(result.input_num_samples / mean_samples_per_frame)
        deviation = abs(result.num_frames - expected_frames)
        if deviation > fc_config.alignment_tolerance_frames:
            print(
                f"{track_id} [{label}]: frame count {result.num_frames} deviates from expected "
                f"{expected_frames} by {deviation} frames (tolerance "
                f"{fc_config.alignment_tolerance_frames})",
                file=sys.stderr,
            )
            return 1
    print(f"alignment check passed for all {len(probes)} probes")

    print("\n--- Cache round-trip ---")
    checked = 0
    max_lowprecision_error = 0.0
    for track_id, label, result, features in probes:
        preprocessing = {
            "target_sample_rate": config.data.target_sample_rate,
            "duration_label": label,
            "num_samples": result.input_num_samples,
        }
        for dtype_name in (config.cache.dtype, fc_config.lowprecision_dtype):
            torch_dtype = getattr(torch, dtype_name)
            key = compute_cache_key(
                CacheKeyInputs(
                    audio_checksum=checksum_by_track_id[track_id],
                    muq_model_id=config.muq.model_id,
                    preprocessing=preprocessing,
                    representation_name=config.cache.representation_name,
                    dtype=dtype_name,
                )
            )
            path = cache_path(cache_dir, key)
            to_save = features.to(torch_dtype).cpu()
            save_atomic(path, to_save, {"track_id": track_id, "duration_label": label})
            loaded_features, _loaded_metadata = load_cache(path)
            if loaded_features.dtype != torch_dtype:
                print(f"Cache round-trip dtype mismatch for {track_id}/{label}/{dtype_name}", file=sys.stderr)
                return 1
            if not torch.equal(loaded_features, to_save):
                print(f"Cache round-trip value mismatch for {track_id}/{label}/{dtype_name}", file=sys.stderr)
                return 1
            checked += 1

            if dtype_name == fc_config.lowprecision_dtype:
                quant_error = (loaded_features.float() - features.float().cpu()).abs().max().item()
                max_lowprecision_error = max(max_lowprecision_error, quant_error)

    print(f"cache round-trip verified exact for {checked} (tensor, dtype) writes")
    print(
        f"max abs error vs float32 for {fc_config.lowprecision_dtype}: "
        f"{max_lowprecision_error:.6f} (tolerance {fc_config.lowprecision_max_abs_error})"
    )
    if max_lowprecision_error > fc_config.lowprecision_max_abs_error:
        print(
            f"{fc_config.lowprecision_dtype} quantization error {max_lowprecision_error:.6f} "
            f"exceeds documented tolerance {fc_config.lowprecision_max_abs_error}",
            file=sys.stderr,
        )
        return 1

    print("\n--- Cache size estimate ---")
    channels = next(iter(channel_values))
    for dtype_name in (config.cache.dtype, fc_config.lowprecision_dtype):
        itemsize = torch.tensor([], dtype=getattr(torch, dtype_name)).element_size()
        bytes_per_second = channels * fps_min * itemsize
        print(f"{dtype_name}: ~{bytes_per_second:,.0f} bytes/audio-second ({itemsize} bytes/value)")

    summary = {
        "probe_tracks": [t.track_id for t in probe_tracks],
        "probes_run": len(probes),
        "feature_channels": channels,
        "frames_per_second_range": [fps_min, fps_max],
        "mean_samples_per_frame": mean_samples_per_frame,
        "cache_entries_verified": checked,
        "lowprecision_max_abs_error": max_lowprecision_error,
    }
    print("\nGate 2 summary:")
    print(json.dumps(summary, indent=2))
    print("\nGate 2 PASSED.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="muq_inversion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Gate 0: environment doctor")
    doctor_parser.add_argument("--config", required=True, type=Path)

    prepare_parser = subparsers.add_parser("prepare-data", help="Gate 1: data integrity")
    prepare_parser.add_argument("--config", required=True, type=Path)

    extract_parser = subparsers.add_parser("extract", help="Gate 2: feature contract")
    extract_parser.add_argument("--config", required=True, type=Path)

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
    if args.command == "extract":
        return cmd_extract(config)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
