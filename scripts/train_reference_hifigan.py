"""Gate 3 tiny-set overfit, using the *unmodified* reference HiFi-GAN implementation
(experiment/reference-hifigan branch) instead of this project's own custom generator/
discriminators/losses.

Purpose: isolate whether this project's poor Gate 3 audio quality originates from (1) our own
vocoder implementation, (2) our training setup/objectives, or (3) something deeper (a limitation
of the tiny-overfit paradigm itself, independent of implementation) -- see
notes/DECISIONS.md#reference-hifigan and notes/SCRATCHPAD.md ("New branch:
experiment/reference-hifigan") for the full fidelity plan and reasoning.

Deliberately NOT threaded through this project's own `config.py`/`train.py`/`cli.py` machinery:
the whole point is running code that is *not* ours, glued to our data with the minimum necessary
code. Reuses this project's `audio.py` (resampling/validation) and `logging_utils.py` (device
selection, no-silent-MPS-fallback) since those are data/infra utilities, not decoder design.

Conditioning: the ground-truth mel spectrogram of the target crop itself (the reference
implementation's own training assumption -- see meldataset.py upstream), computed with the
reference's exact `config_v1.json` mel parameters (22050 Hz, n_fft=1024, hop=256, win=1024,
n_mels=80, fmin=0, fmax=8000). This is intentionally the same "copy synthesis" setup as the
sibling experiment/mel-hifigan-sanity branch, but with the reference's own generator/
discriminators/losses instead of this project's custom MRF generator.

Usage:
    uv run python scripts/train_reference_hifigan.py
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from muq_inversion.audio import load_resample_validate
from muq_inversion.logging_utils import DeviceSelectionError, select_device
from muq_inversion.reference_hifigan.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from muq_inversion.reference_hifigan.generator import Generator
from muq_inversion.reference_hifigan.mel import mel_spectrogram

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- config_v1.json, verbatim (fetched 2026-09-08 from jik876/hifi-gan) ---
SAMPLING_RATE = 22050
N_FFT = 1024
NUM_MELS = 80
HOP_SIZE = 256
WIN_SIZE = 1024
FMIN = 0
FMAX = 8000
UPSAMPLE_RATES = (8, 8, 2, 2)
UPSAMPLE_KERNEL_SIZES = (16, 16, 4, 4)
UPSAMPLE_INITIAL_CHANNEL = 512
RESBLOCK_KERNEL_SIZES = (3, 7, 11)
RESBLOCK_DILATION_SIZES = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
LEARNING_RATE = 2.0e-4
ADAM_BETAS = (0.8, 0.99)
MEL_LOSS_WEIGHT = 45.0  # reference recipe's exact weight, not this project's own choice

# --- deviations required by this experiment's own constraints (see SCRATCHPAD.md plan) ---
SEGMENT_SIZE = 44032  # 172 * 256: nearest exact multiple of HOP_SIZE to 2.0s * 22050 = 44100
STEPS = 15000
SEED = 0
TRACKS = [
    ("pop.00012", "genres/pop/pop.00012.wav"),
    ("disco.00049", "genres/disco/disco.00049.wav"),
]
RUN_NAME = "reference_hifigan"


def _save_atomic(path: Path, state: dict) -> None:
    """Same atomic temp-file + os.replace pattern used throughout this project's checkpoint.py,
    kept minimal here (no full ExperimentConfig/manifest-hash machinery) since this script
    deliberately isn't threaded through this project's own config/train pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    try:
        device = select_device("auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        print(f"DEVICE SELECTION FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"selected device: {device}")

    raw_dir = REPO_ROOT / "data" / "raw" / "gtzan"
    clips = []
    for track_id, relative_path in TRACKS:
        waveform, orig_sr = load_resample_validate(raw_dir / relative_path, SAMPLING_RATE)
        print(f"{track_id}: loaded at native {orig_sr}Hz, resampled to {SAMPLING_RATE}Hz, {len(waveform)} samples")
        clips.append((track_id, torch.from_numpy(waveform)))
        if len(waveform) < SEGMENT_SIZE:
            print(f"{track_id}: waveform shorter than SEGMENT_SIZE={SEGMENT_SIZE}", file=sys.stderr)
            return 1

    print("\n--- Gate-0-style representation diagnostics (reference mel_spectrogram) ---")
    probe_crop = clips[0][1][:SEGMENT_SIZE].unsqueeze(0)
    probe_mel = mel_spectrogram(probe_crop, N_FFT, NUM_MELS, SAMPLING_RATE, HOP_SIZE, WIN_SIZE, FMIN, FMAX)
    print(f"mel shape: {tuple(probe_mel.shape)}")
    print(f"mel dimensionality (channels): {probe_mel.shape[1]}")
    print(f"num_frames: {probe_mel.shape[2]}")
    print(f"frames_per_second: {SAMPLING_RATE / HOP_SIZE:.4f}")
    print(f"samples_per_frame: {SEGMENT_SIZE / probe_mel.shape[2]:.4f}")
    print(f"is_finite: {torch.isfinite(probe_mel).all().item()}")

    generator = Generator(
        num_mels=NUM_MELS,
        upsample_rates=UPSAMPLE_RATES,
        upsample_kernel_sizes=UPSAMPLE_KERNEL_SIZES,
        upsample_initial_channel=UPSAMPLE_INITIAL_CHANNEL,
        resblock_kernel_sizes=RESBLOCK_KERNEL_SIZES,
        resblock_dilation_sizes=RESBLOCK_DILATION_SIZES,
    ).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    n_gen_params = sum(p.numel() for p in generator.parameters())
    print(f"\ngenerator parameters: {n_gen_params:,}")

    total_upsample = 1
    for u in UPSAMPLE_RATES:
        total_upsample *= u
    print(f"total_upsample: {total_upsample} (== hop_size, by the reference's own design)")
    assert total_upsample == HOP_SIZE, "reference config's own upsample product must equal hop_size"
    assert SEGMENT_SIZE % HOP_SIZE == 0, "SEGMENT_SIZE must be an exact multiple of HOP_SIZE"

    optim_g = torch.optim.AdamW(generator.parameters(), LEARNING_RATE, betas=ADAM_BETAS)
    optim_d = torch.optim.AdamW(
        list(msd.parameters()) + list(mpd.parameters()), LEARNING_RATE, betas=ADAM_BETAS
    )

    run_dir = REPO_ROOT / "runs" / RUN_NAME
    checkpoint_dir = run_dir / "checkpoints"

    print("\n--- Training (reference HiFi-GAN, no warmup -- adversarial from step 1) ---")
    mel_loss_history: list[float] = []
    generator.train()
    mpd.train()
    msd.train()
    for step in range(1, STEPS + 1):
        track_id, waveform = clips[step % len(clips)]
        n = waveform.shape[0]
        start = rng.randint(0, n - SEGMENT_SIZE)
        y = waveform[start : start + SEGMENT_SIZE].unsqueeze(0).to(device)

        y_mel = mel_spectrogram(y, N_FFT, NUM_MELS, SAMPLING_RATE, HOP_SIZE, WIN_SIZE, FMIN, FMAX)
        y_g_hat = generator(y_mel)
        y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1), N_FFT, NUM_MELS, SAMPLING_RATE, HOP_SIZE, WIN_SIZE, FMIN, FMAX
        )

        # Discriminator step
        optim_d.zero_grad()
        y_df_r, y_df_g, _, _ = mpd(y.unsqueeze(1), y_g_hat.detach())
        loss_disc_f = discriminator_loss(y_df_r, y_df_g)
        y_ds_r, y_ds_g, _, _ = msd(y.unsqueeze(1), y_g_hat.detach())
        loss_disc_s = discriminator_loss(y_ds_r, y_ds_g)
        loss_disc_all = loss_disc_s + loss_disc_f
        loss_disc_all.backward()
        optim_d.step()

        # Generator step
        optim_g.zero_grad()
        loss_mel = torch.nn.functional.l1_loss(y_mel, y_g_hat_mel)
        _y_df_r2, y_df_g2, fmap_f_r, fmap_f_g = mpd(y.unsqueeze(1), y_g_hat)
        _y_ds_r2, y_ds_g2, fmap_s_r, fmap_s_g = msd(y.unsqueeze(1), y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f = generator_loss(y_df_g2)
        loss_gen_s = generator_loss(y_ds_g2)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * MEL_LOSS_WEIGHT
        loss_gen_all.backward()
        optim_g.step()

        mel_loss_history.append(loss_mel.item())
        if step == 1 or step % 250 == 0 or step == STEPS:
            print(
                f"step {step}/{STEPS}: gen_loss_total={loss_gen_all.item():.4f} "
                f"mel_l1={loss_mel.item():.4f} disc_loss={loss_disc_all.item():.4f} clip={track_id}"
            )

        if step % 1000 == 0 or step == STEPS:
            state = {
                "generator_state_dict": generator.state_dict(),
                "mpd_state_dict": mpd.state_dict(),
                "msd_state_dict": msd.state_dict(),
                "optim_g_state_dict": optim_g.state_dict(),
                "optim_d_state_dict": optim_d.state_dict(),
                "step": step,
                "seed": SEED,
            }
            _save_atomic(checkpoint_dir / f"step_{step:06d}.pt", state)
            _save_atomic(checkpoint_dir / "latest.pt", state)

    initial = sum(mel_loss_history[:10]) / min(10, len(mel_loss_history))
    final = sum(mel_loss_history[-10:]) / min(10, len(mel_loss_history))
    ratio = final / initial if initial > 0 else float("inf")
    print(f"\nmean mel L1 loss (first 10 steps): {initial:.4f}")
    print(f"mean mel L1 loss (last 10 steps): {final:.4f}")
    print(f"final/initial ratio: {ratio:.4f} (PASS threshold, matching this project's Gate 3: <= 0.5)")

    print("\n--- Saving reference/reconstruction/difference audio ---")
    artifacts_dir = REPO_ROOT / "artifacts" / RUN_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        for track_id, waveform in clips:
            eval_crop = waveform[:SEGMENT_SIZE].unsqueeze(0).to(device)
            eval_mel = mel_spectrogram(eval_crop, N_FFT, NUM_MELS, SAMPLING_RATE, HOP_SIZE, WIN_SIZE, FMIN, FMAX)
            reconstruction = generator(eval_mel).squeeze(0).squeeze(0).cpu().numpy()
            reference = eval_crop.squeeze(0).cpu().numpy()
            difference = reference - reconstruction

            sf.write(artifacts_dir / f"{track_id}_reference.wav", reference, SAMPLING_RATE)
            sf.write(artifacts_dir / f"{track_id}_reconstruction.wav", reconstruction, SAMPLING_RATE)
            sf.write(artifacts_dir / f"{track_id}_difference.wav", difference, SAMPLING_RATE)
            print(f"{track_id}: wrote reference/reconstruction/difference to {artifacts_dir}")

    passed = ratio <= 0.5
    if passed:
        print("\nGate 3 (reference HiFi-GAN) PASSED: mel loss decreased substantially.")
        return 0
    print("\nGate 3 (reference HiFi-GAN) FAILED: mel loss did not decrease enough.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
