"""Gate 3: tiny-set overfit training loop (CLAUDE.md Required Validation Gates).

Trains the small HiFi-GAN-style generator, conditioned on frozen MuQ
features, to reconstruct 1-4 fixed clips. Reconstruction-only losses (mel +
multi-resolution STFT) per CLAUDE.md's Decoder Design -- no discriminators
yet. If reconstruction loss does not decrease substantially, this reports
FAIL and the caller must not proceed to Gate 4.
"""

from __future__ import annotations

import dataclasses
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .audio import load_resample_validate
from .checkpoint import build_checkpoint_state, save_checkpoint_atomic
from .config import ExperimentConfig
from .data import TrackRecord, deterministic_unit_interval, read_manifest
from .encoder import FrozenMuQEncoder
from .models.generator import MiniGenerator
from .models.losses import MelSpectrogramLoss, MultiResolutionSTFTLoss


@dataclasses.dataclass
class OverfitClip:
    track_id: str
    waveform: torch.Tensor  # (num_samples,) float32 CPU, resampled to target_sample_rate


def select_overfit_tracks(config: ExperimentConfig, records: list[TrackRecord]) -> list[TrackRecord]:
    usable = [r for r in records if not r.corrupted and r.split == "train"]
    if config.training.overfit_track_ids:
        by_id = {r.track_id: r for r in usable}
        missing = [t for t in config.training.overfit_track_ids if t not in by_id]
        if missing:
            raise ValueError(f"training.overfit_track_ids not found in usable train split: {missing}")
        return [by_id[t] for t in config.training.overfit_track_ids]

    ordered = sorted(
        usable, key=lambda r: deterministic_unit_interval(r.track_id, config.data.split_seed)
    )
    return ordered[: config.training.num_overfit_tracks]


def load_clips(records: list[TrackRecord], raw_dir: Path, target_sample_rate: int) -> list[OverfitClip]:
    clips = []
    for record in records:
        waveform, _orig_sr = load_resample_validate(raw_dir / record.relative_path, target_sample_rate)
        clips.append(OverfitClip(track_id=record.track_id, waveform=torch.from_numpy(waveform)))
    return clips


def sample_training_crop(clip: OverfitClip, crop_samples: int, rng: random.Random) -> torch.Tensor:
    """Random-start crop of exactly `crop_samples` (must be a multiple of the decoder's
    total_upsample so the generator's output length exactly matches this target)."""
    n = clip.waveform.shape[0]
    if n < crop_samples:
        raise ValueError(f"{clip.track_id}: waveform length {n} shorter than crop_samples {crop_samples}")
    start = rng.randint(0, n - crop_samples)
    return clip.waveform[start : start + crop_samples]


def fixed_validation_crop(clip: OverfitClip, crop_samples: int) -> torch.Tensor:
    """Deterministic prefix crop, fixed across runs, for reconstruction/eval audio."""
    return clip.waveform[:crop_samples]


def run_gate3(config: ExperimentConfig, repo_root: Path) -> int:
    manifest_path = repo_root / config.data.manifests_dir / f"gtzan_manifest_{config.name}.csv"
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}. Run prepare-data first.", file=sys.stderr)
        return 1
    records = read_manifest(manifest_path)
    raw_dir = repo_root / config.data.raw_dir

    from .logging_utils import DeviceSelectionError, select_device

    try:
        device = select_device(config.environment.device, config.environment.allow_cpu_fallback)
    except DeviceSelectionError as exc:
        print(f"DEVICE SELECTION FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"selected device: {device}")

    seed = config.environment.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    overfit_records = select_overfit_tracks(config, records)
    if not (1 <= len(overfit_records) <= 4):
        print(
            f"Gate 3 requires 1-4 fixed clips, got {len(overfit_records)}. Adjust "
            "training.num_overfit_tracks or training.overfit_track_ids.",
            file=sys.stderr,
        )
        return 1
    print(f"overfit tracks: {[r.track_id for r in overfit_records]}")

    clips = load_clips(overfit_records, raw_dir, config.data.target_sample_rate)

    encoder = FrozenMuQEncoder(config.muq.model_id, device)

    # Never assume MuQ's channel count (CLAUDE.md MuQ Feature Contract) -- measure it from a
    # real forward pass before building the generator's input projection.
    total_upsample = 1
    for rate in config.decoder.upsample_rates:
        total_upsample *= rate
    probe_crop_samples = config.training.crop_frames * total_upsample
    if clips[0].waveform.shape[0] < probe_crop_samples:
        print(
            f"{clips[0].track_id}: waveform ({clips[0].waveform.shape[0]} samples) shorter than "
            f"crop_samples ({probe_crop_samples}). Reduce training.crop_frames.",
            file=sys.stderr,
        )
        return 1
    probe_waveform = fixed_validation_crop(clips[0], probe_crop_samples).unsqueeze(0).to(device)
    probe_result = encoder.probe(probe_waveform, config.data.target_sample_rate)
    print(f"measured feature_channels: {probe_result.feature_channels}")

    generator = MiniGenerator(
        input_channels=probe_result.feature_channels,
        base_channels=config.decoder.base_channels,
        upsample_rates=config.decoder.upsample_rates,
    ).to(device)
    crop_samples = config.training.crop_frames * generator.total_upsample
    print(f"crop_samples: {crop_samples} ({crop_samples / config.data.target_sample_rate:.2f}s)")

    for clip in clips:
        if clip.waveform.shape[0] < crop_samples:
            print(
                f"{clip.track_id}: waveform ({clip.waveform.shape[0]} samples) shorter than "
                f"crop_samples ({crop_samples}). Reduce training.crop_frames.",
                file=sys.stderr,
            )
            return 1

    mel_loss_fn = MelSpectrogramLoss(
        sample_rate=config.data.target_sample_rate,
        n_fft=config.mel_loss.n_fft,
        hop_length=config.mel_loss.hop_length,
        win_length=config.mel_loss.win_length,
        n_mels=config.mel_loss.n_mels,
        fmin=config.mel_loss.fmin,
        fmax=config.mel_loss.fmax,
    ).to(device)
    stft_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=config.stft_loss.fft_sizes,
        hop_sizes=config.stft_loss.hop_sizes,
        win_lengths=config.stft_loss.win_lengths,
    ).to(device)

    optimizer = torch.optim.Adam(generator.parameters(), lr=config.training.learning_rate)

    run_dir = repo_root / "runs" / config.training.run_name
    checkpoint_dir = run_dir / "checkpoints"

    print("\n--- Training (Gate 3 tiny-set overfit) ---")
    loss_history: list[float] = []
    generator.train()
    for step in range(1, config.training.steps + 1):
        clip = clips[step % len(clips)]
        crop = sample_training_crop(clip, crop_samples, rng).unsqueeze(0).to(device)

        # encoder.encode runs under its own @torch.inference_mode(); clone() here (outside that
        # context) converts the result to a normal, autograd-eligible leaf tensor for the
        # generator's input, without ever requiring gradients through frozen MuQ.
        features = encoder.encode(crop).clone()

        prediction = generator(features)
        mel_l = mel_loss_fn(prediction, crop)
        stft_total, stft_sc, stft_mag = stft_loss_fn(prediction, crop)
        loss = mel_l + stft_total

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if step == 1 or step % config.training.log_every == 0 or step == config.training.steps:
            print(
                f"step {step}/{config.training.steps}: loss={loss.item():.4f} "
                f"mel={mel_l.item():.4f} stft_sc={stft_sc.item():.4f} stft_mag={stft_mag.item():.4f} "
                f"clip={clip.track_id}"
            )

        if step % config.training.checkpoint_every == 0 or step == config.training.steps:
            state = build_checkpoint_state(
                config=config,
                repo_root=repo_root,
                manifest_path=manifest_path,
                generator=generator,
                optimizer=optimizer,
                step=step,
                samples_seen=step * crop_samples,
                best_metric=min(loss_history),
            )
            save_checkpoint_atomic(checkpoint_dir / f"step_{step:06d}.pt", state)
            save_checkpoint_atomic(checkpoint_dir / "latest.pt", state)

    initial_loss = sum(loss_history[:10]) / min(10, len(loss_history))
    final_loss = sum(loss_history[-10:]) / min(10, len(loss_history))
    ratio = final_loss / initial_loss if initial_loss > 0 else float("inf")
    print(f"\nmean loss (first 10 steps): {initial_loss:.4f}")
    print(f"mean loss (last 10 steps): {final_loss:.4f}")
    print(f"final/initial ratio: {ratio:.4f} (PASS threshold: <= {config.training.loss_decrease_ratio})")

    print("\n--- Saving reference/reconstruction/difference audio ---")
    artifacts_dir = repo_root / "artifacts" / config.training.run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        for clip in clips:
            eval_crop = fixed_validation_crop(clip, crop_samples).unsqueeze(0).to(device)
            with torch.inference_mode():
                features = encoder.encode(eval_crop)
            reconstruction = generator(features).squeeze(0).cpu().numpy()
            reference = eval_crop.squeeze(0).cpu().numpy()
            difference = reference - reconstruction

            sf.write(artifacts_dir / f"{clip.track_id}_reference.wav", reference, config.data.target_sample_rate)
            sf.write(
                artifacts_dir / f"{clip.track_id}_reconstruction.wav",
                reconstruction,
                config.data.target_sample_rate,
            )
            sf.write(artifacts_dir / f"{clip.track_id}_difference.wav", difference, config.data.target_sample_rate)
            print(f"{clip.track_id}: wrote reference/reconstruction/difference to {artifacts_dir}")

    passed = ratio <= config.training.loss_decrease_ratio
    if passed:
        print("\nGate 3 PASSED: reconstruction loss decreased substantially.")
        return 0
    print(
        "\nGate 3 FAILED: reconstruction loss did not decrease enough. "
        "Do not proceed to Gate 4 -- diagnose per CLAUDE.md Stop Conditions.",
        file=sys.stderr,
    )
    return 1
