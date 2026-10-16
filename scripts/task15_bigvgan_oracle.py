"""TASK-15 positive control, run before any training:

    original waveform -> official BigVGAN mel -> frozen BigVGAN-v2 -> waveform

on all 8 crops of the N=8 set (a superset of N=2). Writes original/oracle WAVs, mel plots, and
metrics to artifacts/task15_bigvgan_memorization/oracle/.

    uv run python scripts/task15_bigvgan_oracle.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import yaml

from muq_inversion.bigvgan_common import (
    audio_metrics,
    load_bigvgan,
    load_mel_crops,
    mel_l1,
    official_mel,
)
from muq_inversion.logging_utils import select_device
from muq_inversion.models.losses import MultiResolutionSTFTLoss

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/task15_bigvgan_memorization.yaml"
STFT = ((512, 1024, 2048), (50, 120, 240), (240, 600, 1200))  # this project's STFT metric


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    device = select_device(cfg["device_request"], cfg["allow_cpu_fallback"])
    out = REPO_ROOT / "artifacts" / cfg["artifact_dir_name"] / "oracle"
    out.mkdir(parents=True, exist_ok=True)

    model = load_bigvgan(device)
    h = model.h
    print({k: h[k] for k in ("sampling_rate", "num_mels", "n_fft", "hop_size", "win_size", "fmin", "fmax")})
    stft = MultiResolutionSTFTLoss(*STFT)
    crops = load_mel_crops(REPO_ROOT / cfg["manifests"]["8"], h, cfg["num_mel_hops"])

    results = {}
    for c in crops:
        with torch.inference_mode():
            oracle = model(c.target_mel.to(device)).squeeze().cpu()
        original = c.waveform.squeeze()
        assert oracle.shape == original.shape, (oracle.shape, original.shape)
        m = audio_metrics(original, oracle, h, stft)
        m["oracle_mel_vs_target_mel_l1"] = mel_l1(official_mel(oracle.unsqueeze(0), h), c.target_mel)
        m["oracle_peak"] = float(oracle.abs().max())
        results[c.track_id] = m
        sf.write(out / f"{c.track_id}_original.wav", original.numpy(), h.sampling_rate)
        sf.write(out / f"{c.track_id}_bigvgan_oracle.wav", oracle.numpy(), h.sampling_rate)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
        for ax, mel, title in ((axes[0], c.target_mel, "original (official mel)"),
                               (axes[1], official_mel(oracle.unsqueeze(0), h), "BigVGAN oracle (re-analysed)")):
            ax.imshow(mel.squeeze().numpy(), origin="lower", aspect="auto", vmin=-11.5, vmax=2.5)
            ax.set_title(title)
        fig.suptitle(c.track_id)
        fig.tight_layout()
        fig.savefig(out / f"{c.track_id}_mel.png", dpi=110)
        plt.close(fig)
        print(f"{c.track_id:>18}: corr {m['waveform_correlation']:.3f} lag {m['lag_samples']:+d}  "
              f"resynth mel L1 {m['resynthesis_mel_l1']:.3f}  STFT {m['stft_distance']:.3f}")

    keys = ("waveform_correlation", "resynthesis_mel_l1", "stft_distance")
    mean = {k: sum(r[k] for r in results.values()) / len(results) for k in keys}
    print("mean", {k: round(v, 4) for k, v in mean.items()})
    (out / "oracle_metrics.json").write_text(json.dumps({"per_crop": results, "mean": mean,
                                                         "bigvgan_config": dict(h)}, indent=2))


if __name__ == "__main__":
    main()
