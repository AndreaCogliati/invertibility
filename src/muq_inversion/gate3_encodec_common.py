"""Pure, testable helpers for `tasks/TASK-11.md`'s EnCodec continuous-latent controlled scaling
probe (`scripts/train_gate3_encodec_scale_wandb.py`).

This task changes ONLY the target codec relative to `tasks/TASK-05.md`
(`scripts/train_gate3_scale_wandb.py`): DAC `quantized_representation` -> EnCodec quantized
(post-RVQ) continuous latent. Everything else -- input representation (log-mel, from TASK-02's
existing dense cache), manifests, architecture, loss, optimizer, step budget -- is reused
unmodified. `evaluate_prediction`/`aggregate_metrics_across_examples`/`three_way_grad_norms`/
`count_nonfinite` from `gate3_spectrogram_common.py` are reused AS-IS, not reimplemented:
`evaluate_prediction`'s `dac` parameter only ever calls `.decode(...)` on it (duck-typed), and
`FrozenEncodecEncoder.decode` (`encodec_encoder.py`) was deliberately built in TASK-10 to mirror
`FrozenDacEncoder.decode`'s exact interface -- (batch, channels, frames) in, (batch, num_samples)
out -- so passing an `EncodecEncoder` instance in place of `dac` works unchanged, verified below
rather than assumed.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import numpy as np
import torch

from muq_inversion.audio import load_resample_validate
from muq_inversion.dac_shift_sensitivity_common import cosine_similarity_per_frame
from muq_inversion.encodec_encoder import FrozenEncodecEncoder
from muq_inversion.gate3_spectrogram_common import (
    load_examples_from_manifest as load_dac_examples,
)
from muq_inversion.gate4_common import waveform_correlation
from muq_inversion.spectrogram_encoder import LogMelSpectrogramEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class EncodecExampleData:
    track_id: str
    mel: torch.Tensor  # (1, num_frames, n_mels) -- identical to TASK-05's input, reused from the dense cache
    target: torch.Tensor  # (1, num_frames, 128) -- EnCodec quantized latent, frames-first (matches ExampleData.dac's layout)
    waveform: torch.Tensor  # (1, num_samples) -- the raw crop the target was extracted from, kept for the oracle-vs-original audio comparison


def load_examples_from_manifest_encodec(
    manifest_path: Path, device: torch.device, encodec: FrozenEncodecEncoder, sample_rate: int
) -> dict[str, EncodecExampleData]:
    """Reuses TASK-05's exact manifest and dense-cache mel lookup (via `load_dac_examples`, kept
    unmodified) for the input side; computes the EnCodec target fresh from the manifest's own
    `path`/`crop_start_seconds`/`crop_duration_seconds` columns, since TASK-02's dense cache only
    ever stored DAC targets. Raises (does not silently drop) if a track's raw audio crop and its
    cached mel disagree on frame count -- `tasks/TASK-11.md`'s "no off-by-one target frames are
    silently removed" requirement."""
    dac_examples = load_dac_examples(manifest_path, device)

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    examples: dict[str, EncodecExampleData] = {}
    for row in rows:
        track_id = row["track_id"]
        mel = dac_examples[track_id].mel
        waveform, _orig_sr = load_resample_validate(REPO_ROOT / row["path"], sample_rate)
        crop_samples = round(float(row["crop_duration_seconds"]) * sample_rate)
        start = round(float(row["crop_start_seconds"]) * sample_rate)
        end = start + crop_samples
        if end > len(waveform):
            raise ValueError(f"STOP CONDITION: {track_id} crop [{start},{end}) exceeds resampled length {len(waveform)}")
        crop = torch.from_numpy(waveform[start:end].copy()).float().unsqueeze(0).to(device)

        # `.clone()` is required, not cosmetic: `FrozenEncodecEncoder.encode` runs under
        # `torch.inference_mode()`, so its raw output is an inference tensor that PyTorch refuses
        # to use as an autograd target later ("Inference tensors cannot be saved for backward") --
        # unlike TASK-05's DAC targets, which come from `torch.load`-ing TASK-02's dense cache and
        # are therefore already plain tensors by the time training sees them. `.clone()` produces
        # a normal tensor, exactly as PyTorch's own error message for this case recommends.
        target = encodec.encode(crop).transpose(1, 2).clone()  # (1,128,frames) -> (1,frames,128)
        if target.shape[1] != mel.shape[1]:
            raise ValueError(
                f"STOP CONDITION: {track_id} EnCodec target has {target.shape[1]} frames, "
                f"mel input has {mel.shape[1]} -- frame-count mismatch, not silently truncating"
            )
        examples[track_id] = EncodecExampleData(track_id=track_id, mel=mel, target=target, waveform=crop)
    return examples


def find_best_lag_and_align(original: torch.Tensor, reconstruction: torch.Tensor, max_lag: int = 32) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Some codec decoders introduce a small, fixed group delay relative to their input, not just
    a length change -- confirmed empirically for DAC (see `notes/DECISIONS.md#task11-...`: a naive
    lag-0 comparison of `original` against `FrozenDacEncoder`'s own oracle decode gives waveform
    correlation ~0.37, while the true best-aligned lag (8 samples, a systematic decoder delay, not
    noise) gives ~0.97). EnCodec's own decoder was checked the same way and found to need no
    shift (best lag 0). Searches `[-max_lag, max_lag]` by Pearson correlation on 1-D `original`/
    `reconstruction` tensors and returns the aligned (equal-length) pair plus the lag found, so a
    codec-oracle comparison reflects reconstruction quality, not an uncorrected fixed decoder
    delay."""
    orig_np = original.detach().cpu().numpy()
    recon_np = reconstruction.detach().cpu().numpy()
    best_lag, best_corr = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = orig_np[lag:], recon_np[: len(orig_np) - lag]
        else:
            b, a = recon_np[-lag:], orig_np[: len(recon_np) + lag]
        n = min(len(a), len(b))
        if n < 100:
            continue
        corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    if best_lag >= 0:
        aligned_original, aligned_recon = original[best_lag:], reconstruction[: original.shape[-1] - best_lag]
    else:
        aligned_recon, aligned_original = reconstruction[-best_lag:], original[: reconstruction.shape[-1] + best_lag]
    n = min(aligned_original.shape[-1], aligned_recon.shape[-1])
    return aligned_original[:n], aligned_recon[:n], best_lag


def compute_oracle_audio_metrics(
    example: EncodecExampleData, encodec: FrozenEncodecEncoder, stft_loss_fn, device: torch.device
) -> dict[str, float]:
    """`tasks/TASK-11.md`'s "EnCodec Oracle Baseline" requirement: original waveform vs.
    EnCodec-encode-then-decode reconstruction (audio-vs-audio, the frozen codec's own lossy
    reconstruction ceiling -- distinct from any latent-vs-latent comparison, since `example.target`
    is already the encoded latent, not a second independent encoding). Aligned via
    `find_best_lag_and_align` before any metric is computed, since a fixed decoder delay would
    otherwise masquerade as reconstruction error."""
    with torch.no_grad():
        oracle_audio = encodec.decode(example.target.transpose(1, 2)).squeeze(0)
    original = example.waveform.squeeze(0)
    original_n, oracle_n, _lag = find_best_lag_and_align(original.cpu(), oracle_audio.cpu())

    logmel = LogMelSpectrogramEncoder(device)
    mel_a = logmel.encode(original_n.unsqueeze(0).to(device))
    mel_b = logmel.encode(oracle_n.unsqueeze(0).to(device))

    return {
        "waveform_correlation": waveform_correlation(oracle_n, original_n),
        "logmel_cosine_similarity": cosine_similarity_per_frame(mel_a, mel_b, frame_dim=1),
        "stft_distance": float(stft_loss_fn(oracle_n.unsqueeze(0).to(device), original_n.unsqueeze(0).to(device))[0].item()),
    }


def verify_decoder_round_trip(encodec: FrozenEncodecEncoder, waveform_24khz: torch.Tensor) -> bool:
    """`tasks/TASK-11.md`'s Temporal Alignment Validation item 5: "re-encoding and decoding the
    oracle continuous latent produces the expected EnCodec reconstruction." Returns True iff the
    decoder accepts the encoder's own output shape without hidden reshaping/truncation and
    produces a finite, non-empty waveform of comparable length -- a structural check, not a
    quality judgment (reconstruction quality is explicitly out of scope for this task's oracle
    comparison, which uses its own metrics)."""
    latent = encodec.encode(waveform_24khz)
    reconstructed = encodec.decode(latent)
    if not torch.isfinite(reconstructed).all():
        return False
    if reconstructed.shape[0] != waveform_24khz.shape[0]:
        return False
    length_ratio = reconstructed.shape[-1] / waveform_24khz.shape[-1]
    return 0.9 <= length_ratio <= 1.1
