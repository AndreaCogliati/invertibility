"""tasks/TASK-07.md: does DAC's `quantized_representation` (and the pre-quantizer encoder
output) uniquely determine, or ignore, small sub-frame time shifts and global polarity of its
input waveform? A pre-training diagnostic, on the frozen DAC encoder alone -- no MuQ, no
translator, no training loop, no checkpoint touched.

## Method summary (full spec: tasks/TASK-07.md)

20 GTZAN val-split tracks (2/genre, deterministic, `select_stratified_tracks`), 2 crops/track
(2.0s each) = 40 crops. For each crop:
  - shift sensitivity: cosine similarity + codebook agreement between the canonical crop and the
    crop re-extracted at +/-{1,2,4,8,16,32,64,128,256}-sample offsets, both pre- and
    post-quantizer;
  - polarity sensitivity: same comparisons for `x` vs. `-x` (canonical alignment only);
  - the same shift sweep repeated for log-mel and MuQ (the two candidate translator inputs), for
    the load-bearing "is DAC uniquely fragile, or do the inputs shift just as much" comparison.
A random cross-track-pair baseline (~40 pairs, `sample_cross_track_pairs`, seed=0) is the
"chance" floor every similarity number is read against.

## Shift extraction: real-audio margin, not circular wraparound

Each crop is drawn from a padded buffer with `MAX_SHIFT` samples of genuine surrounding audio on
each side (crop starts are chosen with a much larger margin, so this is always available). A
shift-by-k crop is a literal re-slice of that buffer, `k` samples later/earlier -- not
`torch.roll` on the same tensor. Both the canonical and every shifted crop are therefore full
`CROP_SAMPLES`-length, real-audio (never zero-padded or wrapped) inputs, so every one of them
produces the same number of valid frames from each encoder and no edge frames need to be dropped
for invalidity; `frames_dropped` is reported as 0 throughout and this is documented, not silently
assumed -- see `tasks/TASK-07.md` step 2's "say how many frames were dropped" requirement. Frame
index i of the shifted crop is then compared directly against frame index i of the canonical
crop: both cover the same *position in the frame sequence*, but the shifted one's frame sees
audio content offset by k samples -- precisely the ambiguity a shift-invariant translator input
cannot resolve, which is what this diagnostic measures.

## Stop conditions handled

- Pre-quantizer output: confirmed available without any `transformers` monkeypatch --
  `DacModel.encoder`/`DacModel.quantizer` are already public submodules whose composition
  reproduces `DacModel.encode()` exactly (verified once, see
  `FrozenDacEncoder.encode_stages`'s docstring). No fallback needed.
- Non-finite crops: checked per-crop after loading; a failing crop is dropped, logged, and
  excluded from every downstream computation (never silently skipped without a record).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from muq_inversion.audio import load_resample_validate
from muq_inversion.dac_encoder import FrozenDacEncoder
from muq_inversion.dac_shift_sensitivity_common import (
    TrackSpec,
    codebook_agreement,
    cosine_similarity_per_frame,
    sample_cross_track_pairs,
    select_stratified_tracks,
)
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.logging_utils import select_device
from muq_inversion.spectrogram_encoder import LogMelSpectrogramEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/gtzan_manifest_smoke.csv"
RAW_AUDIO_ROOT = REPO_ROOT / "data/raw/gtzan"
FIGURE_DIR = REPO_ROOT / "artifacts/task07_dac_shift_sensitivity"

SAMPLE_RATE = 24000
DAC_MODEL_ID = "descript/dac_24khz"
MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"

TRACKS_PER_GENRE = 2
CROP_SECONDS = 2.0
CROP_SAMPLES = round(CROP_SECONDS * SAMPLE_RATE)
CROP_START_SECONDS = [5.0, 15.0]  # 2 crops/track
SHIFT_MAGNITUDES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MAX_SHIFT = max(SHIFT_MAGNITUDES)
N_RANDOM_PAIRS = 40
SEED = 0
ASYMMETRY_THRESHOLD = 0.05  # abs(mean(+k) - mean(-k)) above this -> report both, not just average


def crop_id(track_id: str, crop_start: float) -> str:
    return f"{track_id}_t{crop_start:.1f}s"


def load_padded_crops(tracks: list[TrackSpec]) -> tuple[dict[str, torch.Tensor], dict[str, str], list[str]]:
    """Returns (crop_id -> padded buffer of length CROP_SAMPLES+2*MAX_SHIFT, crop_id -> track_id,
    list of dropped crop_ids with reason already printed) -- `tasks/TASK-07.md`'s finite-value
    stop condition #2: a failing crop is dropped and logged, not fatal."""
    buffers: dict[str, torch.Tensor] = {}
    track_of_crop: dict[str, str] = {}
    dropped: list[str] = []
    for track in tracks:
        waveform, _orig_sr = load_resample_validate(RAW_AUDIO_ROOT / track.relative_path, SAMPLE_RATE)
        for crop_start in CROP_START_SECONDS:
            cid = crop_id(track.track_id, crop_start)
            start = round(crop_start * SAMPLE_RATE)
            lo, hi = start - MAX_SHIFT, start + CROP_SAMPLES + MAX_SHIFT
            if lo < 0 or hi > len(waveform):
                raise ValueError(
                    f"STOP CONDITION: {track.track_id} crop at {crop_start}s needs samples "
                    f"[{lo},{hi}) but track has only {len(waveform)} resampled samples"
                )
            buf = waveform[lo:hi]
            if not np.all(np.isfinite(buf)):
                print(f"dropping {cid}: non-finite samples in padded buffer")
                dropped.append(cid)
                continue
            buffers[cid] = torch.from_numpy(buf.copy()).float()
            track_of_crop[cid] = track.track_id
    return buffers, track_of_crop, dropped


def slice_shifted(buffer: torch.Tensor, shift: int) -> torch.Tensor:
    """buffer: (CROP_SAMPLES + 2*MAX_SHIFT,). shift=0 is the canonical crop."""
    start = MAX_SHIFT + shift
    return buffer[start : start + CROP_SAMPLES]


def summarize_direction_pair(pos: float, neg: float) -> dict[str, float | bool]:
    asymmetric = abs(pos - neg) > ASYMMETRY_THRESHOLD
    return {"plus": pos, "minus": neg, "mean": (pos + neg) / 2, "asymmetric": asymmetric}


def run_shift_sweep_dac(
    dac: FrozenDacEncoder, buffers: dict[str, torch.Tensor], crop_ids: list[str]
) -> dict[int, dict[str, dict[str, float | bool]]]:
    """For each shift magnitude, batches canonical + shift(+k) + shift(-k) across all crops in
    one forward pass per magnitude (cheap: DAC on 2.0s clips is fast, no gradients). Returns
    {magnitude: {"quantized": summary, "pre_quantizer": summary, "codebook_stage0": summary,
    "codebook_mean_all_stages": summary}}."""
    results: dict[int, dict[str, dict[str, float | bool]]] = {}
    canonical = torch.stack([slice_shifted(buffers[c], 0) for c in crop_ids])
    canonical_stages = dac.encode_stages(canonical)

    for k in SHIFT_MAGNITUDES:
        plus = torch.stack([slice_shifted(buffers[c], k) for c in crop_ids])
        minus = torch.stack([slice_shifted(buffers[c], -k) for c in crop_ids])
        plus_stages = dac.encode_stages(plus)
        minus_stages = dac.encode_stages(minus)

        q_plus = cosine_similarity_per_frame(canonical_stages.quantized_representation, plus_stages.quantized_representation, frame_dim=2)
        q_minus = cosine_similarity_per_frame(canonical_stages.quantized_representation, minus_stages.quantized_representation, frame_dim=2)
        pre_plus = cosine_similarity_per_frame(canonical_stages.pre_quantizer_encoding, plus_stages.pre_quantizer_encoding, frame_dim=2)
        pre_minus = cosine_similarity_per_frame(canonical_stages.pre_quantizer_encoding, minus_stages.pre_quantizer_encoding, frame_dim=2)
        agree_plus = codebook_agreement(canonical_stages.audio_codes, plus_stages.audio_codes)
        agree_minus = codebook_agreement(canonical_stages.audio_codes, minus_stages.audio_codes)

        results[k] = {
            "quantized": summarize_direction_pair(q_plus, q_minus),
            "pre_quantizer": summarize_direction_pair(pre_plus, pre_minus),
            "codebook_stage0": summarize_direction_pair(agree_plus["stage0_agreement"], agree_minus["stage0_agreement"]),
            "codebook_mean_all_stages": summarize_direction_pair(
                agree_plus["mean_agreement_all_stages"], agree_minus["mean_agreement_all_stages"]
            ),
            "frames_dropped": 0,
        }
    return results


def run_shift_sweep_generic(encode_fn, buffers: dict[str, torch.Tensor], crop_ids: list[str], frame_dim: int) -> dict[int, dict[str, float | bool]]:
    """Same shift sweep as `run_shift_sweep_dac`, for a single-tensor-output encoder (log-mel,
    MuQ) -- no quantizer/codebook stage."""
    results: dict[int, dict[str, float | bool]] = {}
    canonical = torch.stack([slice_shifted(buffers[c], 0) for c in crop_ids])
    canonical_feat = encode_fn(canonical)
    for k in SHIFT_MAGNITUDES:
        plus = torch.stack([slice_shifted(buffers[c], k) for c in crop_ids])
        minus = torch.stack([slice_shifted(buffers[c], -k) for c in crop_ids])
        plus_feat = encode_fn(plus)
        minus_feat = encode_fn(minus)
        sim_plus = cosine_similarity_per_frame(canonical_feat, plus_feat, frame_dim=frame_dim)
        sim_minus = cosine_similarity_per_frame(canonical_feat, minus_feat, frame_dim=frame_dim)
        results[k] = summarize_direction_pair(sim_plus, sim_minus)
        results[k]["frames_dropped"] = 0
    return results


def run_polarity_dac(dac: FrozenDacEncoder, buffers: dict[str, torch.Tensor], crop_ids: list[str]) -> dict[str, float]:
    canonical = torch.stack([slice_shifted(buffers[c], 0) for c in crop_ids])
    pos_stages = dac.encode_stages(canonical)
    neg_stages = dac.encode_stages(-canonical)
    agreement = codebook_agreement(pos_stages.audio_codes, neg_stages.audio_codes)
    return {
        "quantized_cosine_similarity": cosine_similarity_per_frame(pos_stages.quantized_representation, neg_stages.quantized_representation, frame_dim=2),
        "pre_quantizer_cosine_similarity": cosine_similarity_per_frame(pos_stages.pre_quantizer_encoding, neg_stages.pre_quantizer_encoding, frame_dim=2),
        "codebook_stage0_agreement": agreement["stage0_agreement"],
        "codebook_mean_agreement_all_stages": agreement["mean_agreement_all_stages"],
    }


def run_random_pair_baseline(
    dac: FrozenDacEncoder, buffers: dict[str, torch.Tensor], crop_ids: list[str], track_of_crop: dict[str, str]
) -> dict[str, float]:
    pairs = sample_cross_track_pairs(crop_ids, track_of_crop, n_pairs=N_RANDOM_PAIRS, seed=SEED)
    canonical = {c: slice_shifted(buffers[c], 0) for c in crop_ids}
    left = torch.stack([canonical[a] for a, _b in pairs])
    right = torch.stack([canonical[b] for _a, b in pairs])
    left_stages = dac.encode_stages(left)
    right_stages = dac.encode_stages(right)
    agreement = codebook_agreement(left_stages.audio_codes, right_stages.audio_codes)
    return {
        "n_pairs": len(pairs),
        "quantized_cosine_similarity_mean": cosine_similarity_per_frame(left_stages.quantized_representation, right_stages.quantized_representation, frame_dim=2),
        "pre_quantizer_cosine_similarity_mean": cosine_similarity_per_frame(left_stages.pre_quantizer_encoding, right_stages.pre_quantizer_encoding, frame_dim=2),
        "codebook_stage0_agreement_mean": agreement["stage0_agreement"],
        "codebook_mean_agreement_all_stages_mean": agreement["mean_agreement_all_stages"],
    }


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    device = select_device("auto", allow_cpu_fallback=False)
    print(f"device: {device}")

    t0 = time.time()

    tracks = select_stratified_tracks(MANIFEST_PATH, split="val", tracks_per_genre=TRACKS_PER_GENRE)
    print(f"{len(tracks)} tracks selected ({TRACKS_PER_GENRE}/genre, val split)")

    buffers, track_of_crop, dropped = load_padded_crops(tracks)
    crop_ids = sorted(buffers)
    print(f"{len(crop_ids)} crops loaded ({len(dropped)} dropped: {dropped})")

    dac = FrozenDacEncoder(DAC_MODEL_ID, device)
    n_codebooks = dac.model.config.n_codebooks
    print(f"DAC n_codebooks={n_codebooks} (measured, not assumed)")

    buffers = {c: b.to(device) for c, b in buffers.items()}

    print("running DAC shift sweep...")
    dac_shift = run_shift_sweep_dac(dac, buffers, crop_ids)

    print("running DAC polarity sweep...")
    dac_polarity = run_polarity_dac(dac, buffers, crop_ids)

    print("running random-pair baseline...")
    random_baseline = run_random_pair_baseline(dac, buffers, crop_ids, track_of_crop)

    print("running log-mel shift sweep...")
    logmel = LogMelSpectrogramEncoder(device)
    logmel_shift = run_shift_sweep_generic(logmel.encode, buffers, crop_ids, frame_dim=1)

    print("running MuQ shift sweep...")
    muq = FrozenMuQEncoder(MUQ_MODEL_ID, device)
    muq_shift = run_shift_sweep_generic(muq.encode, buffers, crop_ids, frame_dim=1)

    elapsed = time.time() - t0
    print(f"finished: {elapsed:.1f}s")

    report = {
        "sample_rate": SAMPLE_RATE,
        "crop_samples": CROP_SAMPLES,
        "crop_seconds": CROP_SECONDS,
        "shift_magnitudes": SHIFT_MAGNITUDES,
        "n_codebooks": n_codebooks,
        "asymmetry_threshold": ASYMMETRY_THRESHOLD,
        "crops": {
            "used": crop_ids,
            "dropped": dropped,
            "track_of_crop": track_of_crop,
            "tracks": [{"track_id": t.track_id, "genre": t.genre} for t in tracks],
            "crop_start_seconds": CROP_START_SECONDS,
        },
        "dac_shift_sensitivity": dac_shift,
        "dac_polarity_sensitivity": dac_polarity,
        "random_pair_baseline": random_baseline,
        "logmel_shift_sensitivity": logmel_shift,
        "muq_shift_sensitivity": muq_shift,
        "elapsed_seconds": elapsed,
    }
    out_path = FIGURE_DIR / "shift_sensitivity_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")

    plot_figure(dac_shift, logmel_shift, muq_shift, random_baseline)


def plot_figure(dac_shift, logmel_shift, muq_shift, random_baseline) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    shifts = SHIFT_MAGNITUDES

    dac_q = [dac_shift[k]["quantized"]["mean"] for k in shifts]
    dac_pre = [dac_shift[k]["pre_quantizer"]["mean"] for k in shifts]
    logmel_vals = [logmel_shift[k]["mean"] for k in shifts]
    muq_vals = [muq_shift[k]["mean"] for k in shifts]

    ax1.plot(shifts, dac_q, "o-", label="DAC quantized_representation", color="tab:red")
    ax1.plot(shifts, dac_pre, "s-", label="DAC pre-quantizer encoding", color="tab:orange")
    ax1.plot(shifts, logmel_vals, "^-", label="log-mel (translator input)", color="tab:blue")
    ax1.plot(shifts, muq_vals, "d-", label="MuQ last_hidden_state (translator input)", color="tab:green")
    ax1.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax1.axhline(random_baseline["quantized_cosine_similarity_mean"], color="tab:red", linewidth=1.0, linestyle="--", label="DAC random-pair baseline")

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Shift magnitude (samples, log2 scale)")
    ax1.set_ylabel("Mean cosine similarity to unshifted (canonical) crop")
    ax1.set_title("Shift sensitivity: DAC latents vs. translator input representations")
    ax1.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out_path = FIGURE_DIR / "shift_sensitivity.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
