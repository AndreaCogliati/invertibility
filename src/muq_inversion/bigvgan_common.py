"""TASK-15: MuQ -> BigVGAN-v2 mel -> frozen BigVGAN-v2 memorization test.

Kept separate from the EnCodec experiments (TASK-11..14). The BigVGAN code is NVIDIA's own,
fetched at a pinned revision by `scripts/fetch_bigvgan.py` into
`third_party/bigvgan_v2_24khz_100band_256x/`; nothing here reimplements it. Mel spectrograms come
from that directory's `meldataset.mel_spectrogram`, called with the checkpoint's own
`config.json` hyperparameters (`model.h`), so the representation is exactly the one the vocoder
was trained on.

Waveform conditioning follows the checkpoint's training data path (`config.json`
`normalize_volume: true`; `meldataset.MelDataset.__getitem__` with `fine_tuning=False`): each
full track is peak-normalized to 0.95 with `librosa.util.normalize` before cropping. The README's
inference snippet skips this step; the training path is what defines the checkpoint's expected
input distribution.
"""

from __future__ import annotations

import csv
import dataclasses
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

from muq_inversion.audio import load_resample_validate
from muq_inversion.gate3_encodec_common import find_best_lag_and_align
from muq_inversion.gate4_common import waveform_correlation

REPO_ROOT = Path(__file__).resolve().parents[2]
BIGVGAN_DIR = REPO_ROOT / "third_party" / "bigvgan_v2_24khz_100band_256x"
PEAK_NORMALIZE_TARGET = 0.95  # meldataset.MelDataset: `normalize(audio) * 0.95`


def _import_official_modules(bigvgan_dir: Path):
    if not (bigvgan_dir / "bigvgan.py").exists():
        raise FileNotFoundError(f"{bigvgan_dir} missing -- run `uv run python scripts/fetch_bigvgan.py` first")
    if str(bigvgan_dir) not in sys.path:
        sys.path.insert(0, str(bigvgan_dir))
    import bigvgan
    import meldataset

    return bigvgan, meldataset


def load_bigvgan(device: torch.device, bigvgan_dir: Path = BIGVGAN_DIR):
    """Official loading path (README): `BigVGAN.from_pretrained(..., use_cuda_kernel=False)`,
    `remove_weight_norm()`, `eval()`. Frozen: no parameter requires grad."""
    bigvgan, _ = _import_official_modules(bigvgan_dir)
    model = bigvgan.BigVGAN.from_pretrained(str(bigvgan_dir), use_cuda_kernel=False)
    model.remove_weight_norm()
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def official_mel(waveform: torch.Tensor, h, bigvgan_dir: Path = BIGVGAN_DIR) -> torch.Tensor:
    """waveform: (B, num_samples) -> log-mel (B, num_mels, num_samples // hop_size), via the
    checkpoint's own `meldataset.mel_spectrogram` with its own hyperparameters. Computed on CPU
    so targets don't depend on the accelerator's STFT implementation."""
    _, meldataset = _import_official_modules(bigvgan_dir)
    return meldataset.mel_spectrogram(
        waveform.detach().float().cpu(), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
        h.win_size, h.fmin, h.fmax,
    )


def bigvgan_log_mel_floor(bigvgan_dir: Path = BIGVGAN_DIR) -> float:
    """Smallest value the official preprocessing can produce: `spectral_normalize_torch` applied
    to zero magnitude, i.e. log(clip_val * C) from `dynamic_range_compression_torch`'s defaults
    (log(1e-5) for this checkpoint)."""
    _, meldataset = _import_official_modules(bigvgan_dir)
    return float(meldataset.spectral_normalize_torch(torch.zeros(1)))


def clamp_to_mel_floor(mel: torch.Tensor, floor: float) -> tuple[torch.Tensor, int]:
    """Returns the mel clamped to `floor` (for BigVGAN input) and how many bins were raised."""
    return mel.clamp(min=floor), int((mel < floor).sum())


@dataclasses.dataclass(frozen=True)
class MelCrop:
    track_id: str
    genre: str
    crop_start_seconds: float
    waveform: torch.Tensor  # (1, num_hops * hop_size), peak-normalized track, then cropped
    target_mel: torch.Tensor  # (1, num_mels, num_hops)


def load_mel_crops(manifest_path: Path, h, num_hops: int) -> list[MelCrop]:
    """One crop per manifest row, `num_hops * h.hop_size` samples long starting at the row's
    `crop_start_seconds`, so the official mel has exactly `num_hops` frames and BigVGAN's output
    is exactly as long as the crop (no trimming anywhere)."""
    num_samples = num_hops * h.hop_size
    crops = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            track, _ = load_resample_validate(REPO_ROOT / row["path"], h.sampling_rate)
            track = librosa.util.normalize(track) * PEAK_NORMALIZE_TARGET
            start = round(float(row["crop_start_seconds"]) * h.sampling_rate)
            if start + num_samples > len(track):
                raise ValueError(f"{row['track_id']}: crop [{start}, {start + num_samples}) exceeds {len(track)} samples")
            wav = torch.from_numpy(np.ascontiguousarray(track[start : start + num_samples])).float().unsqueeze(0)
            mel = official_mel(wav, h)
            if mel.shape[-1] != num_hops:
                raise ValueError(f"{row['track_id']}: mel has {mel.shape[-1]} frames, expected {num_hops}")
            crops.append(
                MelCrop(
                    track_id=row["track_id"], genre=row["genre"],
                    crop_start_seconds=float(row["crop_start_seconds"]), waveform=wav, target_mel=mel,
                )
            )
    return crops


def shuffled_across_pairing(track_ids: list[str]) -> dict[str, str]:
    """Deterministic derangement: each crop takes the MuQ features of the next crop in manifest
    order (cyclic). Never pairs a crop with itself for N >= 2."""
    if len(track_ids) < 2:
        raise ValueError("shuffled-across control needs at least 2 crops")
    return {tid: track_ids[(i + 1) % len(track_ids)] for i, tid in enumerate(track_ids)}


def mel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean absolute difference between two log-mel spectrograms (natural-log units, the same
    L1-on-log-mel form BigVGAN itself is trained with)."""
    return float((a.detach().cpu() - b.detach().cpu()).abs().mean())


def audio_metrics(reference: torch.Tensor, candidate: torch.Tensor, h, stft_loss_fn) -> dict[str, float]:
    """reference/candidate: 1-D waveforms of equal length. Waveform correlation is computed after
    lag alignment (`find_best_lag_and_align`); resynthesis mel L1 compares the official mel of each
    full waveform; STFT distance is this project's multi-resolution STFT metric."""
    ref, cand = reference.detach().float().cpu(), candidate.detach().float().cpu()
    ref_a, cand_a, lag = find_best_lag_and_align(ref, cand)
    with torch.no_grad():
        stft = float(stft_loss_fn(cand.unsqueeze(0), ref.unsqueeze(0))[0])
    return {
        "waveform_correlation": waveform_correlation(cand_a, ref_a),
        "lag_samples": lag,
        "resynthesis_mel_l1": mel_l1(official_mel(cand.unsqueeze(0), h), official_mel(ref.unsqueeze(0), h)),
        "stft_distance": stft,
    }
