"""Gate 4: short pilot (CLAUDE.md Required Validation Gates, Evaluation and Controls).

Trains on a larger deterministic subset than Gate 3, validates checkpoint/
resume equivalence, and trains three conditioning-control variants
(shuffled across examples, shuffled in time, constant/zero) to confirm real
MuQ conditioning materially outperforms all of them. If it doesn't, CLAUDE.md
Stop Conditions say to assume conditioning collapse/leakage until disproven.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .checkpoint import (
    build_checkpoint_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
)
from .config import ExperimentConfig
from .data import read_manifest
from .encoder import FrozenMuQEncoder
from .logging_utils import DeviceSelectionError, select_device
from .models.generator import MiniGenerator
from .models.losses import MelSpectrogramLoss, MultiResolutionSTFTLoss
from .train import (
    OverfitClip,
    fixed_validation_crop,
    load_clips,
    sample_training_crop,
    select_tracks,
)

CONDITIONING_MODES = ("shuffled_across", "shuffled_time", "constant_zero")


def _build_conditioning(
    mode: str,
    encoder: FrozenMuQEncoder,
    target_clip: OverfitClip,
    target_crop_device: torch.Tensor,
    other_clips: list[OverfitClip],
    crop_samples: int,
    rng: random.Random,
) -> torch.Tensor:
    """Compute the conditioning features for one training step under `mode`.

    "correct": encode the clip's own crop (real per-frame temporal conditioning).
    "shuffled_across": encode a different clip's crop entirely -- tests whether the
      decoder distinguishes examples by conditioning at all.
    "shuffled_time": encode the clip's own crop, then permute frame order -- tests
      whether the decoder uses *temporal* structure or just per-clip averages.
    "constant_zero": zero out the conditioning tensor -- tests whether the decoder
      can reconstruct without any MuQ information (memorization/leakage check).
    """
    if mode == "shuffled_across":
        candidates = [c for c in other_clips if c.track_id != target_clip.track_id] or [target_clip]
        source_clip = candidates[rng.randrange(len(candidates))]
        source_crop = sample_training_crop(source_clip, crop_samples, rng)
        source_crop = source_crop.unsqueeze(0).to(target_crop_device.device)
    else:
        source_crop = target_crop_device

    features = encoder.encode(source_crop).clone()

    if mode == "shuffled_time":
        perm = torch.randperm(features.shape[1])
        features = features[:, perm, :]
    elif mode == "constant_zero":
        features = torch.zeros_like(features)
    return features


def _make_generator_and_optimizer(
    config: ExperimentConfig, feature_channels: int, device: torch.device, seed: int, learning_rate: float
) -> tuple[MiniGenerator, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    generator = MiniGenerator(
        input_channels=feature_channels,
        base_channels=config.decoder.base_channels,
        upsample_rates=config.decoder.upsample_rates,
    ).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    return generator, optimizer


def _make_losses(config: ExperimentConfig, device: torch.device) -> tuple[MelSpectrogramLoss, MultiResolutionSTFTLoss]:
    mel = MelSpectrogramLoss(
        sample_rate=config.data.target_sample_rate,
        n_fft=config.mel_loss.n_fft,
        hop_length=config.mel_loss.hop_length,
        win_length=config.mel_loss.win_length,
        n_mels=config.mel_loss.n_mels,
        fmin=config.mel_loss.fmin,
        fmax=config.mel_loss.fmax,
    ).to(device)
    stft = MultiResolutionSTFTLoss(
        fft_sizes=config.stft_loss.fft_sizes,
        hop_sizes=config.stft_loss.hop_sizes,
        win_lengths=config.stft_loss.win_lengths,
    ).to(device)
    return mel, stft


def _train_steps(
    *,
    generator: MiniGenerator,
    optimizer: torch.optim.Optimizer,
    mel_loss_fn: MelSpectrogramLoss,
    stft_loss_fn: MultiResolutionSTFTLoss,
    encoder: FrozenMuQEncoder,
    clips: list[OverfitClip],
    crop_samples: int,
    mode: str,
    rng: random.Random,
    start_step: int,
    end_step: int,
    log_every: int,
    log_label: str,
    device: torch.device,
) -> list[float]:
    loss_history: list[float] = []
    generator.train()
    for step in range(start_step, end_step + 1):
        clip = clips[step % len(clips)]
        target_crop = sample_training_crop(clip, crop_samples, rng).unsqueeze(0).to(device)

        if mode == "correct":
            features = encoder.encode(target_crop).clone()
        else:
            features = _build_conditioning(mode, encoder, clip, target_crop, clips, crop_samples, rng)

        prediction = generator(features)
        mel_l = mel_loss_fn(prediction, target_crop)
        stft_total, _stft_sc, _stft_mag = stft_loss_fn(prediction, target_crop)
        loss = mel_l + stft_total

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if step == start_step or step % log_every == 0 or step == end_step:
            print(f"[{log_label}] step {step}/{end_step}: loss={loss.item():.4f} clip={clip.track_id}")
    return loss_history


def _tail_mean(losses: list[float], n: int = 10) -> float:
    return sum(losses[-n:]) / min(n, len(losses))


def run_gate4(config: ExperimentConfig, repo_root: Path) -> int:
    manifest_path = repo_root / config.data.manifests_dir / f"gtzan_manifest_{config.name}.csv"
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}. Run prepare-data first.", file=sys.stderr)
        return 1
    records = read_manifest(manifest_path)
    raw_dir = repo_root / config.data.raw_dir
    pilot_cfg = config.pilot

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

    if not (0 < pilot_cfg.resume_test_at < pilot_cfg.steps):
        print("pilot.resume_test_at must be strictly between 0 and pilot.steps.", file=sys.stderr)
        return 1

    pilot_records = select_tracks(
        records,
        split_seed=config.data.split_seed,
        num_tracks=pilot_cfg.num_tracks,
        explicit_track_ids=pilot_cfg.track_ids,
    )
    if len(pilot_records) < 2:
        print(
            "Gate 4 needs at least 2 tracks (to test shuffled-across-examples conditioning). "
            "Increase pilot.num_tracks.",
            file=sys.stderr,
        )
        return 1
    print(f"pilot tracks: {[r.track_id for r in pilot_records]}")

    clips = load_clips(pilot_records, raw_dir, config.data.target_sample_rate)

    encoder = FrozenMuQEncoder(config.muq.model_id, device)

    total_upsample = 1
    for rate in config.decoder.upsample_rates:
        total_upsample *= rate
    crop_samples = pilot_cfg.crop_frames * total_upsample
    for clip in clips:
        if clip.waveform.shape[0] < crop_samples:
            print(
                f"{clip.track_id}: waveform ({clip.waveform.shape[0]} samples) shorter than "
                f"crop_samples ({crop_samples}). Reduce pilot.crop_frames.",
                file=sys.stderr,
            )
            return 1

    probe_waveform = fixed_validation_crop(clips[0], crop_samples).unsqueeze(0).to(device)
    feature_channels = encoder.probe(probe_waveform, config.data.target_sample_rate).feature_channels
    print(f"measured feature_channels: {feature_channels}")

    run_dir = repo_root / "runs" / pilot_cfg.run_name
    checkpoint_dir = run_dir / "checkpoints"

    # ---- Checkpoint/resume equivalence (correct conditioning only) ----
    print("\n--- Checkpoint/resume equivalence check ---")
    resume_at = pilot_cfg.resume_test_at

    mel_a, stft_a = _make_losses(config, device)
    gen_a, opt_a = _make_generator_and_optimizer(config, feature_channels, device, seed, pilot_cfg.learning_rate)
    rng_a = random.Random(seed)
    uninterrupted_losses = _train_steps(
        generator=gen_a, optimizer=opt_a, mel_loss_fn=mel_a, stft_loss_fn=stft_a, encoder=encoder,
        clips=clips, crop_samples=crop_samples, mode="correct", rng=rng_a,
        start_step=1, end_step=pilot_cfg.steps, log_every=pilot_cfg.log_every,
        log_label="uninterrupted", device=device,
    )

    mel_b, stft_b = _make_losses(config, device)
    gen_b, opt_b = _make_generator_and_optimizer(config, feature_channels, device, seed, pilot_cfg.learning_rate)
    rng_b = random.Random(seed)
    first_half_losses = _train_steps(
        generator=gen_b, optimizer=opt_b, mel_loss_fn=mel_b, stft_loss_fn=stft_b, encoder=encoder,
        clips=clips, crop_samples=crop_samples, mode="correct", rng=rng_b,
        start_step=1, end_step=resume_at, log_every=pilot_cfg.log_every,
        log_label="pre-checkpoint", device=device,
    )
    # MPS kernels are not guaranteed bit-exact across separate invocations (CLAUDE.md Stop
    # Conditions anticipate "expected nondeterminism"), so this uses the same relative tolerance
    # as the resume check below rather than a byte-exact comparison.
    max_predeterminism_diff = max(
        abs(x - y) / max(abs(x), 1e-6)
        for x, y in zip(uninterrupted_losses[:resume_at], first_half_losses, strict=True)
    )
    determinism_ok = max_predeterminism_diff <= pilot_cfg.resume_tolerance
    print(
        f"pre-checkpoint determinism match (same seed, same steps): {determinism_ok} "
        f"(max relative diff {max_predeterminism_diff:.4f}, tolerance {pilot_cfg.resume_tolerance})"
    )

    checkpoint_path = checkpoint_dir / "resume_test_checkpoint.pt"
    state_b = build_checkpoint_state(
        config=config, repo_root=repo_root, manifest_path=manifest_path,
        generator=gen_b, optimizer=opt_b, step=resume_at,
        samples_seen=resume_at * crop_samples, best_metric=min(first_half_losses),
        extra={"crop_rng_state": rng_b.getstate()},
    )
    save_checkpoint_atomic(checkpoint_path, state_b)

    # Fresh model/optimizer from a *different* seed, to prove the checkpoint load -- not the
    # initialization -- determines the resumed run's trajectory.
    gen_c, opt_c = _make_generator_and_optimizer(
        config, feature_channels, device, seed + 1, pilot_cfg.learning_rate
    )
    loaded = load_checkpoint(checkpoint_path)
    gen_c.load_state_dict(loaded["generator_state_dict"])
    opt_c.load_state_dict(loaded["optimizer_state_dict"])
    restore_rng_state(loaded)
    rng_c = random.Random()
    rng_c.setstate(loaded["crop_rng_state"])

    resumed_second_half_losses = _train_steps(
        generator=gen_c, optimizer=opt_c, mel_loss_fn=mel_b, stft_loss_fn=stft_b, encoder=encoder,
        clips=clips, crop_samples=crop_samples, mode="correct", rng=rng_c,
        start_step=resume_at + 1, end_step=pilot_cfg.steps, log_every=pilot_cfg.log_every,
        log_label="resumed", device=device,
    )

    uninterrupted_second_half = uninterrupted_losses[resume_at:]
    max_rel_diff = max(
        abs(x - y) / max(abs(x), 1e-6)
        for x, y in zip(uninterrupted_second_half, resumed_second_half_losses, strict=True)
    )
    print(
        f"max relative loss difference (resumed vs uninterrupted, second half): {max_rel_diff:.4f} "
        f"(tolerance {pilot_cfg.resume_tolerance}; not expected to be bit-exact on MPS)"
    )
    resume_ok = determinism_ok and max_rel_diff <= pilot_cfg.resume_tolerance
    print(f"checkpoint/resume equivalence: {'PASSED' if resume_ok else 'FAILED'}")

    # ---- Conditioning controls ----
    print("\n--- Conditioning controls ---")
    control_losses: dict[str, list[float]] = {"correct": uninterrupted_losses}
    for mode in CONDITIONING_MODES:
        mel_m, stft_m = _make_losses(config, device)
        gen_m, opt_m = _make_generator_and_optimizer(
            config, feature_channels, device, seed, pilot_cfg.learning_rate
        )
        rng_m = random.Random(seed)
        control_losses[mode] = _train_steps(
            generator=gen_m, optimizer=opt_m, mel_loss_fn=mel_m, stft_loss_fn=stft_m, encoder=encoder,
            clips=clips, crop_samples=crop_samples, mode=mode, rng=rng_m,
            start_step=1, end_step=pilot_cfg.steps, log_every=pilot_cfg.log_every,
            log_label=mode, device=device,
        )

    print("\n--- Control comparison (mean loss, last 10 steps) ---")
    final_losses = {mode: _tail_mean(losses) for mode, losses in control_losses.items()}
    for mode, value in final_losses.items():
        print(f"{mode}: {value:.4f}")

    correct_final = final_losses["correct"]
    controls_outperformed: dict[str, bool] = {}
    for mode in CONDITIONING_MODES:
        required_max = final_losses[mode] * (1 - pilot_cfg.control_margin)
        controls_outperformed[mode] = correct_final <= required_max
        verdict = "materially better" if controls_outperformed[mode] else "NOT materially better"
        print(
            f"correct ({correct_final:.4f}) vs {mode} ({final_losses[mode]:.4f}): {verdict} "
            f"(needs <= {required_max:.4f} for margin {pilot_cfg.control_margin})"
        )
    controls_ok = all(controls_outperformed.values())
    if not controls_ok:
        print(
            "\nWARNING: real conditioning does not materially outperform all controls -- per "
            "CLAUDE.md Stop Conditions, assume conditioning collapse/leakage until disproven.",
            file=sys.stderr,
        )

    print("\n--- Saving reference/reconstruction audio (correct-conditioning model) ---")
    artifacts_dir = repo_root / "artifacts" / pilot_cfg.run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    gen_a.eval()
    with torch.no_grad():
        for clip in clips:
            eval_crop = fixed_validation_crop(clip, crop_samples).unsqueeze(0).to(device)
            features = encoder.encode(eval_crop)
            reconstruction = gen_a(features).squeeze(0).cpu().numpy()
            reference = eval_crop.squeeze(0).cpu().numpy()
            sf.write(artifacts_dir / f"{clip.track_id}_reference.wav", reference, config.data.target_sample_rate)
            sf.write(
                artifacts_dir / f"{clip.track_id}_reconstruction.wav",
                reconstruction,
                config.data.target_sample_rate,
            )

    final_state = build_checkpoint_state(
        config=config, repo_root=repo_root, manifest_path=manifest_path,
        generator=gen_a, optimizer=opt_a, step=pilot_cfg.steps,
        samples_seen=pilot_cfg.steps * crop_samples, best_metric=min(uninterrupted_losses),
    )
    save_checkpoint_atomic(checkpoint_dir / "latest.pt", final_state)

    summary = {
        "pilot_tracks": [r.track_id for r in pilot_records],
        "determinism_ok": determinism_ok,
        "resume_max_relative_diff": max_rel_diff,
        "resume_ok": resume_ok,
        "final_losses": final_losses,
        "controls_outperformed": controls_outperformed,
        "controls_ok": controls_ok,
    }
    print("\nGate 4 summary:")
    print(json.dumps(summary, indent=2))

    if resume_ok and controls_ok:
        print("\nGate 4 PASSED.")
        return 0
    print("\nGate 4 FAILED. Do not proceed to Gate 5 -- diagnose per CLAUDE.md Stop Conditions.", file=sys.stderr)
    return 1
