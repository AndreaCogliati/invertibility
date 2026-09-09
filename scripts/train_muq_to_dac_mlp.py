"""Proof-of-concept: predict DAC latents from MuQ latents with a tiny framewise MLP.

    MuQ latent (25 Hz, 1024 ch)
        -> nearest-neighbor repeat 3x (temporal alignment, see measured frame rates below)
        -> framewise MLP (Linear[1024,2048] -> GELU -> Linear[2048,2048] -> GELU -> Linear[2048,1024])
        -> predicted DAC latent (75 Hz, 1024 ch)

Both MuQ and DAC (`descript/dac_24khz`) are kept frozen and in eval mode. Training objective is
pure latent-space regression: MSE(predicted, true DAC latent), optionally plus a small
cosine-similarity term -- no waveform loss, no HiFi-GAN, no GAN, no discriminators, no
transformer, no temporal modeling beyond the fixed nearest-neighbor repeat above.

Standalone script, not threaded through `config.py`/`train.py`/`cli.py`: this experiment's
architecture (frozen-encoder-to-frozen-encoder latent regression) is structurally unrelated to
this project's HiFi-GAN-style decoder pipeline, so gluing it into the same config/train
machinery would add indirection without reuse. Reuses only `muq_inversion.audio` (data loading)
and `muq_inversion.logging_utils` (device selection), matching the precedent set by
`scripts/train_reference_hifigan.py`.

Findings and full write-up: notes/DECISIONS.md#muq-to-dac-mlp, notes/SCRATCHPAD.md.
"""

from __future__ import annotations

import os
import tempfile
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

REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_RATE = 24000  # both MuQ and DAC (descript/dac_24khz) operate natively at 24 kHz
CROP_SECONDS = 2.0
CROP_SAMPLES = int(CROP_SECONDS * SAMPLE_RATE)  # 48000, matches Gate 3's own crop length

MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
DAC_MODEL_ID = "descript/dac_24khz"

HIDDEN_CHANNELS = 2048
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)  # plain regression, no adversarial training -- no collapse risk to guard against
COSINE_LOSS_WEIGHT = 0.1
STEPS = 3000
LOG_EVERY = 250
CHECKPOINT_EVERY = 1000
SEED = 0

TRACKS = [
    ("pop.00012", REPO_ROOT / "data/raw/gtzan/genres/pop/pop.00012.wav"),
    ("disco.00049", REPO_ROOT / "data/raw/gtzan/genres/disco/disco.00049.wav"),
]
RUN_NAME = "muq_to_dac_mlp"
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

    if dac_probe.frames_per_second % muq_probe.frames_per_second != 0:
        ratio_check = dac_probe.frames_per_second / muq_probe.frames_per_second
        print(f"WARNING: DAC/MuQ frame-rate ratio {ratio_check:.4f} is not an exact integer")
    frame_ratio = round(dac_probe.frames_per_second / muq_probe.frames_per_second)
    assert muq_probe.num_frames * frame_ratio == dac_probe.num_frames, (
        f"temporal alignment requires an exact integer repeat factor: "
        f"muq_frames={muq_probe.num_frames} * {frame_ratio} != dac_frames={dac_probe.num_frames}"
    )
    print(
        f"temporal alignment strategy: nearest-neighbor repeat each MuQ frame {frame_ratio}x "
        f"({muq_probe.frames_per_second:.4f} Hz -> {dac_probe.frames_per_second:.4f} Hz), "
        f"{muq_probe.num_frames} -> {muq_probe.num_frames * frame_ratio} frames "
        f"(target has {dac_probe.num_frames})"
    )

    print("\n--- Precomputing MuQ (input) and DAC (target) latents for both clips ---")
    muq_latents: dict[str, torch.Tensor] = {}
    dac_targets: dict[str, torch.Tensor] = {}
    for track_id, waveform in clips.items():
        muq_feat = muq.encode(waveform)  # (1, num_frames, 1024)
        dac_feat = dac.encode(waveform)  # (1, 1024, num_frames)
        assert muq_feat.shape[1] * frame_ratio == dac_feat.shape[-1], (
            f"{track_id}: frame count mismatch after measuring geometry -- "
            f"muq {muq_feat.shape} dac {dac_feat.shape}"
        )
        # .clone() is required, not just .detach(): encode() runs under torch.inference_mode(),
        # and inference tensors cannot be used in autograd computation even after detaching.
        muq_latents[track_id] = muq_feat.detach().clone()
        dac_targets[track_id] = dac_feat.transpose(1, 2).detach().clone()  # -> (1, num_frames, 1024)
        print(f"{track_id}: muq {tuple(muq_feat.shape)} -> dac target {tuple(dac_targets[track_id].shape)}")

    print("\n--- Building framewise MLP ---")
    mlp = FramewiseMLP(
        input_channels=muq_probe.feature_channels,
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=dac_probe.latent_channels,
    ).to(device)
    param_count = sum(p.numel() for p in mlp.parameters())
    print(f"MLP parameters: {param_count:,}")

    optimizer = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()

    print("\n--- Training (pure latent-space regression, no adversarial/waveform loss) ---")
    mse_history: list[float] = []
    cos_history: list[float] = []
    for step in range(1, STEPS + 1):
        track_id, _ = TRACKS[step % len(TRACKS)]
        muq_feat = muq_latents[track_id]
        target = dac_targets[track_id]

        repeated = muq_feat.repeat_interleave(frame_ratio, dim=1)  # (1, num_frames*ratio, 1024)
        predicted = mlp(repeated)

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
                "model_state_dict": mlp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "hidden_channels": HIDDEN_CHANNELS,
                    "learning_rate": LEARNING_RATE,
                    "adam_betas": ADAM_BETAS,
                    "cosine_loss_weight": COSINE_LOSS_WEIGHT,
                    "frame_ratio": frame_ratio,
                },
            }
            _save_atomic(RUN_DIR / "checkpoints" / f"step_{step:06d}.pt", state)
            _save_atomic(RUN_DIR / "checkpoints" / "latest.pt", state)

    mean_first = sum(mse_history[:10]) / 10
    mean_last = sum(mse_history[-10:]) / 10
    print(f"\nmean mse (first 10 steps): {mean_first:.6f}")
    print(f"mean mse (last 10 steps): {mean_last:.6f}")
    print(f"final/initial ratio: {mean_last / mean_first:.4f}")
    print(f"final cosine similarity (last 10 steps): {sum(cos_history[-10:]) / 10:.4f}")

    print("\n--- Decoding predicted DAC latents through the frozen DAC decoder ---")
    for track_id, waveform in clips.items():
        muq_feat = muq_latents[track_id]
        target = dac_targets[track_id]
        repeated = muq_feat.repeat_interleave(frame_ratio, dim=1)
        with torch.no_grad():
            predicted = mlp(repeated)

        predicted_for_decode = predicted.transpose(1, 2)  # (1, 1024, num_frames)
        target_for_decode = target.transpose(1, 2)

        reconstructed = dac.decode(predicted_for_decode).squeeze(0).squeeze(0)
        dac_oracle = dac.decode(target_for_decode).squeeze(0).squeeze(0)
        reference = waveform.squeeze(0)

        n = min(len(reference), len(reconstructed), len(dac_oracle))
        difference = reference[:n] - reconstructed[:n].cpu()

        _write_wav(ARTIFACT_DIR / f"{track_id}_reference.wav", reference[:n], SAMPLE_RATE)
        _write_wav(ARTIFACT_DIR / f"{track_id}_reconstruction.wav", reconstructed[:n], SAMPLE_RATE)
        _write_wav(ARTIFACT_DIR / f"{track_id}_dac_oracle.wav", dac_oracle[:n], SAMPLE_RATE)
        _write_wav(ARTIFACT_DIR / f"{track_id}_difference.wav", difference, SAMPLE_RATE)
        print(f"{track_id}: wrote reference/reconstruction/dac_oracle/difference to {ARTIFACT_DIR}")

    print("\nGate 3-style pure-overfit run complete.")


if __name__ == "__main__":
    main()
