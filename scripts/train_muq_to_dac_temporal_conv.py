"""Smallest-possible temporal model for predicting DAC latents from MuQ latents.

    MuQ sequence (25 Hz, 1024 ch)
        -> nearest-neighbor repeat 3x (same temporal alignment as the framewise MLP experiment)
        -> TemporalConvNet: 3x [Conv1D(k=5, 1024->1024) -> GELU, residual] -> Conv1D(k=5, 1024->1024)
        -> predicted DAC latent (75 Hz, 1024 ch)

Single-variable change from `experiment/muq-to-dac-mlp`: everything (input, target, optimizer,
loss, step budget, clips, crop) is identical except the framewise MLP's pointwise `Linear` layers
are replaced by `Conv1D(k=5)` layers that can see neighboring (repeated) frames. This isolates
whether *local temporal context specifically* -- not a bigger or differently-shaped network in
general -- explains any of the framewise MLP's residual error.

No transformers, no attention, no GAN, no waveform loss, no HiFi-GAN, no DAC fine-tuning -- MuQ,
the DAC encoder, and the DAC decoder are all frozen throughout.

Standalone script, not threaded through `config.py`/`train.py`/`cli.py`, matching the precedent
set by `scripts/train_reference_hifigan.py` and `scripts/train_muq_to_dac_mlp.py`.

Findings and full write-up: notes/DECISIONS.md#muq-to-dac-temporal-conv, notes/SCRATCHPAD.md.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

from muq_inversion.audio import load_resample_validate
from muq_inversion.dac_encoder import FrozenDacEncoder
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.logging_utils import DeviceSelectionError, select_device
from muq_inversion.models.framewise_mlp import FramewiseMLP
from muq_inversion.models.temporal_conv import TemporalConvNet

REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_RATE = 24000  # both MuQ and DAC (descript/dac_24khz) operate natively at 24 kHz
CROP_SECONDS = 2.0
CROP_SAMPLES = int(CROP_SECONDS * SAMPLE_RATE)  # 48000, matches Gate 3's own crop length

MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
DAC_MODEL_ID = "descript/dac_24khz"

# Identical to experiment/muq-to-dac-mlp's hyperparameters, so this is a single-variable
# (architecture-only) comparison against that experiment's results.
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
COSINE_LOSS_WEIGHT = 0.1
STEPS = 3000
LOG_EVERY = 250
CHECKPOINT_EVERY = 1000
SEED = 0

# Sibling experiment's checkpoint, reused (not retrained) purely to regenerate a matching
# "MLP reconstruction" WAV in this experiment's own artifacts dir for a direct, self-contained
# comparison. Still present on disk (git-ignored, survives branch switches).
MLP_CHECKPOINT_PATH = REPO_ROOT / "runs/muq_to_dac_mlp/checkpoints/latest.pt"
MLP_HIDDEN_CHANNELS = 2048

TRACKS = [
    ("pop.00012", REPO_ROOT / "data/raw/gtzan/genres/pop/pop.00012.wav"),
    ("disco.00049", REPO_ROOT / "data/raw/gtzan/genres/disco/disco.00049.wav"),
]
RUN_NAME = "muq_to_dac_temporal_conv"
RUN_DIR = REPO_ROOT / "runs" / RUN_NAME
ARTIFACT_DIR = REPO_ROOT / "artifacts" / RUN_NAME


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


def _write_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().numpy().astype(np.float32), sample_rate)


def _measure_receptive_field_empirically(
    model: TemporalConvNet, num_frames: int, channels: int, device: torch.device
) -> int:
    """Perturb a single frame in the middle of a random input sequence and count how many
    output frames change (beyond floating-point noise). Verifies the analytical
    `receptive_field_frames` property against actual model behavior, per CLAUDE.md's
    "measure, don't assume" principle -- rather than trusting the kernel-size arithmetic alone."""
    torch.manual_seed(SEED)
    base_input = torch.randn(1, num_frames, channels, device=device)
    perturbed_input = base_input.clone()
    mid = num_frames // 2
    perturbed_input[0, mid, :] += 10.0  # large perturbation, easily distinguishable from noise

    with torch.no_grad():
        base_output = model(base_input)
        perturbed_output = model(perturbed_input)

    diff = (base_output - perturbed_output).abs().amax(dim=-1).squeeze(0)  # (num_frames,)
    affected = (diff > 1e-4).nonzero(as_tuple=True)[0]
    return int(affected.max().item() - affected.min().item() + 1) if len(affected) > 0 else 0


def main() -> None:
    torch.manual_seed(SEED)
    try:
        device = select_device(requested="auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")

    print("\n--- Loading and cropping clips ---")
    clips: dict[str, torch.Tensor] = {}
    for track_id, path in TRACKS:
        waveform, orig_sr = load_resample_validate(path, SAMPLE_RATE)
        if len(waveform) < CROP_SAMPLES:
            raise SystemExit(f"{track_id}: clip too short ({len(waveform)} < {CROP_SAMPLES} samples)")
        start = (len(waveform) - CROP_SAMPLES) // 2  # deterministic centered crop
        crop = waveform[start : start + CROP_SAMPLES]
        clips[track_id] = torch.from_numpy(crop).float().unsqueeze(0)  # (1, CROP_SAMPLES)
        print(f"{track_id}: orig_sr={orig_sr} crop_samples={len(crop)}")

    print("\n--- Loading frozen encoders ---")
    muq = FrozenMuQEncoder(MUQ_MODEL_ID, device)
    dac = FrozenDacEncoder(DAC_MODEL_ID, device)

    print("\n--- Gate-0-style diagnostics: measuring frame rates and required alignment ---")
    probe_waveform = clips[TRACKS[0][0]]
    muq_probe = muq.probe(probe_waveform, SAMPLE_RATE)
    dac_probe = dac.probe(probe_waveform, SAMPLE_RATE)
    print(f"MuQ:  {muq_probe}")
    print(f"DAC:  {dac_probe}")

    frame_ratio = round(dac_probe.frames_per_second / muq_probe.frames_per_second)
    assert muq_probe.num_frames * frame_ratio == dac_probe.num_frames, (
        f"temporal alignment requires an exact integer repeat factor: "
        f"muq_frames={muq_probe.num_frames} * {frame_ratio} != dac_frames={dac_probe.num_frames}"
    )
    print(
        f"temporal alignment strategy (unchanged from the framewise-MLP experiment): "
        f"nearest-neighbor repeat each MuQ frame {frame_ratio}x, "
        f"{muq_probe.num_frames} -> {muq_probe.num_frames * frame_ratio} frames"
    )

    print("\n--- Precomputing MuQ (input) and DAC (target) latents for both clips ---")
    muq_latents: dict[str, torch.Tensor] = {}
    dac_targets: dict[str, torch.Tensor] = {}
    for track_id, waveform in clips.items():
        muq_feat = muq.encode(waveform)  # (1, num_frames, 1024)
        dac_feat = dac.encode(waveform)  # (1, 1024, num_frames)
        assert muq_feat.shape[1] * frame_ratio == dac_feat.shape[-1], (
            f"{track_id}: frame count mismatch -- muq {muq_feat.shape} dac {dac_feat.shape}"
        )
        # .clone() is required, not just .detach(): encode() runs under torch.inference_mode(),
        # and inference tensors cannot be used in autograd computation even after detaching.
        muq_latents[track_id] = muq_feat.detach().clone()
        dac_targets[track_id] = dac_feat.transpose(1, 2).detach().clone()  # -> (1, num_frames, 1024)
        print(f"{track_id}: muq {tuple(muq_feat.shape)} -> dac target {tuple(dac_targets[track_id].shape)}")

    print("\n--- Building TemporalConvNet ---")
    model = TemporalConvNet(channels=muq_probe.feature_channels).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"TemporalConvNet parameters: {param_count:,}")
    print(f"receptive field (analytical, at DAC's 75 Hz timeline): {model.receptive_field_frames} frames")
    print(
        f"receptive field (analytical, translated to MuQ frames): "
        f"{-(-model.receptive_field_frames // frame_ratio)} frames "
        f"(ceil({model.receptive_field_frames}/{frame_ratio}))"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()

    print("\n--- Training (pure latent-space regression, no adversarial/waveform loss) ---")
    train_start = time.perf_counter()
    mse_history: list[float] = []
    cos_history: list[float] = []
    for step in range(1, STEPS + 1):
        track_id, _ = TRACKS[step % len(TRACKS)]
        muq_feat = muq_latents[track_id]
        target = dac_targets[track_id]

        repeated = muq_feat.repeat_interleave(frame_ratio, dim=1)  # (1, num_frames*ratio, 1024)
        predicted = model(repeated)

        mse = mse_loss_fn(predicted, target)
        cos_sim = nn.functional.cosine_similarity(predicted, target, dim=-1).mean()
        cos_loss = 1.0 - cos_sim
        total_loss = mse + COSINE_LOSS_WEIGHT * cos_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        mse_history.append(mse.item())
        cos_history.append(cos_sim.item())

        if step == 1 or step % LOG_EVERY == 0 or step == STEPS:
            print(
                f"step {step}/{STEPS}: mse={mse.item():.6f} cos_sim={cos_sim.item():.4f} "
                f"total_loss={total_loss.item():.6f} clip={track_id}"
            )

        if step % CHECKPOINT_EVERY == 0 or step == STEPS:
            state = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "learning_rate": LEARNING_RATE,
                    "adam_betas": ADAM_BETAS,
                    "cosine_loss_weight": COSINE_LOSS_WEIGHT,
                    "frame_ratio": frame_ratio,
                    "kernel_size": model.conv1.kernel_size[0],
                },
            }
            _save_atomic(RUN_DIR / "checkpoints" / f"step_{step:06d}.pt", state)
            _save_atomic(RUN_DIR / "checkpoints" / "latest.pt", state)
    train_elapsed = time.perf_counter() - train_start

    mean_first = sum(mse_history[:10]) / 10
    mean_last = sum(mse_history[-10:]) / 10
    print(f"\ntraining time: {train_elapsed:.2f}s ({STEPS} steps, {train_elapsed / STEPS * 1000:.2f} ms/step)")
    print(f"mean mse (first 10 steps): {mean_first:.6f}")
    print(f"mean mse (last 10 steps): {mean_last:.6f}")
    print(f"final/initial ratio: {mean_last / mean_first:.4f}")
    print(f"final cosine similarity (last 10 steps): {sum(cos_history[-10:]) / 10:.4f}")

    print("\n--- Verifying receptive field empirically (perturbation test, not just formula) ---")
    empirical_rf = _measure_receptive_field_empirically(
        model, num_frames=dac_probe.num_frames, channels=muq_probe.feature_channels, device=device
    )
    print(
        f"empirical receptive field (DAC-timeline frames affected by a single perturbed frame): "
        f"{empirical_rf} (analytical: {model.receptive_field_frames})"
    )

    print("\n--- Loading sibling MLP checkpoint for a direct, self-contained comparison ---")
    mlp_available = MLP_CHECKPOINT_PATH.exists()
    mlp: FramewiseMLP | None = None
    if mlp_available:
        mlp_state = torch.load(MLP_CHECKPOINT_PATH, map_location=device, weights_only=False)
        mlp = FramewiseMLP(
            input_channels=muq_probe.feature_channels,
            hidden_channels=MLP_HIDDEN_CHANNELS,
            output_channels=dac_probe.latent_channels,
        ).to(device)
        mlp.load_state_dict(mlp_state["model_state_dict"])
        mlp.eval()
        print(f"loaded {MLP_CHECKPOINT_PATH} (step {mlp_state['step']})")
    else:
        print(f"WARNING: {MLP_CHECKPOINT_PATH} not found -- skipping MLP comparison audio")

    print("\n--- Decoding through the frozen DAC decoder ---")
    for track_id, waveform in clips.items():
        muq_feat = muq_latents[track_id]
        target = dac_targets[track_id]
        repeated = muq_feat.repeat_interleave(frame_ratio, dim=1)
        with torch.no_grad():
            temporal_conv_pred = model(repeated)
            mlp_pred = mlp(repeated) if mlp is not None else None

        reference = waveform.squeeze(0)
        temporal_conv_recon = dac.decode(temporal_conv_pred.transpose(1, 2)).squeeze(0).squeeze(0)
        dac_oracle = dac.decode(target.transpose(1, 2)).squeeze(0).squeeze(0)

        n = min(len(reference), len(temporal_conv_recon), len(dac_oracle))
        difference = reference[:n] - temporal_conv_recon[:n].cpu()

        _write_wav(ARTIFACT_DIR / f"{track_id}_reference.wav", reference[:n], SAMPLE_RATE)
        _write_wav(ARTIFACT_DIR / f"{track_id}_temporal_conv_reconstruction.wav", temporal_conv_recon[:n], SAMPLE_RATE)
        _write_wav(ARTIFACT_DIR / f"{track_id}_dac_oracle.wav", dac_oracle[:n], SAMPLE_RATE)
        _write_wav(ARTIFACT_DIR / f"{track_id}_difference.wav", difference, SAMPLE_RATE)

        if mlp_pred is not None:
            mlp_recon = dac.decode(mlp_pred.transpose(1, 2)).squeeze(0).squeeze(0)
            n2 = min(n, len(mlp_recon))
            _write_wav(ARTIFACT_DIR / f"{track_id}_mlp_reconstruction.wav", mlp_recon[:n2], SAMPLE_RATE)

        print(f"{track_id}: wrote reference/temporal_conv_reconstruction/dac_oracle/difference"
              f"{'/mlp_reconstruction' if mlp_pred is not None else ''} to {ARTIFACT_DIR}")

    print("\nGate 3-style pure-overfit run complete.")


if __name__ == "__main__":
    main()
