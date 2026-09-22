"""TASK-11: EnCodec continuous-latent controlled scaling probe.

Changes exactly one variable relative to TASK-05 (`scripts/train_gate3_scale_wandb.py`): the
target codec, DAC `quantized_representation` -> EnCodec quantized (post-RVQ) continuous latent
(`facebook/encodec_24khz`, the checkpoint validated in TASK-10, at its maximum bandwidth (24.0
kbps, all 32 codebooks) to mirror DAC's own full-RVQ-depth convention). Input representation
(log-mel, from TASK-02's existing dense cache), manifests (`data/manifests/gate3_scale_{2,8,32}.csv`,
reused unmodified and verified nested), architecture (`SpectrogramToDacTranslator`/
`TemporalConvNet`, unchanged except the mechanically-necessary `dac_channels=128` instead of
1024), loss (MSE + 0.1*cosine), optimizer (Adam, lr=1e-3, betas=(0.9,0.999)), step budget (20000,
patience 5000 at 1e-5 min delta), and random seed (0) are all reused byte-for-byte from TASK-05 --
see `notes/DECISIONS.md#task05-gate3-scaling` for the source-of-truth values pulled from that
task's own report.json, per this task's "use the repository record as the source of truth"
instruction.

**Scope**: N in {2, 8, 32} only, per this task's explicit "do not run N=128/512" instruction.

## EnCodec representation and data-flow documentation (per this task's explicit requirement)

- Checkpoint: `facebook/encodec_24khz`. Input sample rate 24000 Hz, mono (1-channel). Frame rate
  75 Hz (measured, identical to DAC's own 75 Hz and to the mel input's own 75 Hz -- no resampling
  needed, verified: a 48000-sample/2.0s crop produces exactly 150 frames for mel, DAC, and
  EnCodec alike). Continuous-latent dimensionality: 128 (both pre- and post-quantizer -- notably
  smaller than DAC's 1024). Tensor orientation: `EncodecEncoder.decode()` expects
  (batch, 128, num_frames) -- channels-first, matching `FrozenDacEncoder.decode()`'s own
  convention; the translator itself, matching TASK-05's `SpectrogramToDacTranslator`, operates
  frames-first (batch, num_frames, 128) and is transposed at the decode boundary only, identical
  to TASK-05's own DAC handling. Bandwidth: 24.0 kbps (all 32 codebooks, `codebook_size=1024`,
  matching DAC's 32/1024 exactly) -- the maximum this checkpoint supports, chosen to mirror TASK-05's
  DAC convention of always using full RVQ depth (`n_quantizers=None`), not a reduced bandwidth.
  Representation: the **post-quantizer** (quantized) continuous latent -- `FrozenEncodecEncoder.encode()`
  -- the same representation whose shift stability was measured in TASK-10, and the one
  `EncodecModel.decode()` actually consumes (the natural "predict then decode" target, matching
  why TASK-01 chose DAC's own `quantized_representation` for the identical reason).

Data flow:

```
waveform -> frozen EnCodec encoder -> continuous target latent (z_encodec, post-RVQ)
input representation (log-mel) -> learned translator -> predicted continuous EnCodec latent
    -> frozen EnCodec decoder -> reconstructed waveform
```

## Parameter count: an unavoidable dimensional change, not a capacity increase

`SpectrogramToDacTranslator(input_channels=128, dac_channels=128)` (this task) has far *fewer*
parameters than TASK-05's `SpectrogramToDacTranslator(input_channels=128, dac_channels=1024)` --
the `dac_channels` argument sets both the input projection's output width and the entire
`TemporalConvNet`'s channel width throughout (three residual `Conv1d(k=5)` layers), so a smaller
target dimensionality mechanically shrinks the whole network, not just its output layer. This is
reported explicitly in the run report (`parameter_count_comparison`) rather than left implicit --
if anything, this works *against* any success at N=32, since fewer parameters is a strictly
smaller capacity budget than TASK-05's DAC run had, not a confound in EnCodec's favor.

## Instrumentation

Reuses `three_way_grad_norms`/`count_nonfinite`/`evaluate_prediction`/`aggregate_metrics_across_examples`/
`rank_examples_by_metric` from `gate3_spectrogram_common.py` (all built for DAC, all untouched --
`evaluate_prediction`'s `dac` parameter only ever calls `.decode(...)`, verified to accept a
`FrozenEncodecEncoder` in tests/test_gate3_encodec_common.py's duck-typing test). Tracks best
checkpoint by BOTH the TASK-05 criterion (mean training-set MSE, primary, used for early stopping)
and by mean waveform correlation to oracle on the fixed 2-track subset (secondary, per this task's
"best checkpoint by waveform correlation, if already supported" instruction) -- TASK-06 showed
final-step evaluation can be misleading given this pipeline's characteristic training-trajectory
turbulence, so both the best and final checkpoints are evaluated and compared, not just the final.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

import wandb
from muq_inversion.dac_encoder import FrozenDacEncoder
from muq_inversion.dac_shift_sensitivity_common import cosine_similarity_per_frame
from muq_inversion.encodec_encoder import FrozenEncodecEncoder
from muq_inversion.gate3_encodec_common import (
    compute_oracle_audio_metrics,
    find_best_lag_and_align,
    load_examples_from_manifest_encodec,
    verify_decoder_round_trip,
)
from muq_inversion.gate3_spectrogram_common import (
    TRACKS,
    aggregate_metrics_across_examples,
    count_nonfinite,
    evaluate_prediction,
    rank_examples_by_metric,
    three_way_grad_norms,
)
from muq_inversion.gate4_common import waveform_correlation
from muq_inversion.logging_utils import (
    DeviceSelectionError,
    capture_environment_manifest,
    select_device,
)
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.models.spectrogram_translator import SpectrogramToDacTranslator
from muq_inversion.spectrogram_encoder import N_MELS, LogMelSpectrogramEncoder
from muq_inversion.wandb_logging import init_run, log_artifact_file, make_run_name

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 24000
DAC_MODEL_ID = "descript/dac_24khz"
ENCODEC_MODEL_ID = "facebook/encodec_24khz"
ENCODEC_BANDWIDTH = 24.0  # kbps, all 32 codebooks -- mirrors DAC's full-RVQ-depth convention
ENCODEC_CHANNELS = 128

# Identical to TASK-05 (source of truth: notes/DECISIONS.md#task05-gate3-scaling /
# artifacts/task05_gate3_scale/n{N}/report.json) -- only LEVELS is restricted per this task's scope.
LEVELS = [2, 8, 32]
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
COSINE_LOSS_WEIGHT = 0.1
SEED = 0
MAX_STEPS = 20000
VAL_EVERY = 250
LOG_EVERY = 1000
PATIENCE_STEPS = 5000
PATIENCE_MIN_DELTA = 1e-5

STFT_FFT_SIZES = (512, 1024, 2048)
STFT_HOP_SIZES = (50, 120, 240)
STFT_WIN_LENGTHS = (240, 600, 1200)

FIXED_AUDIO_SUBSET = TRACKS  # pop.00012/disco.00049, present in every gate3_scale_N manifest

TASK05_PARAMETER_COUNT = 15_863_808  # artifacts/task05_gate3_scale/n2/report.json (identical at every N)


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


def gate0_static_validation(encodec: FrozenEncodecEncoder, device: torch.device) -> None:
    """`tasks/TASK-11.md`'s Gate 0: verify manifests, tensor shapes, decoder interface, oracle
    reconstruction, deterministic extraction -- before any training starts."""
    print("Gate 0: static validation...")
    for n in LEVELS:
        manifest_path = REPO_ROOT / f"data/manifests/gate3_scale_{n}.csv"
        if not manifest_path.exists():
            raise SystemExit(f"STOP CONDITION: {manifest_path} missing -- cannot reuse TASK-05's manifests")
    track_sets = {}
    for n in LEVELS:
        with open(REPO_ROOT / f"data/manifests/gate3_scale_{n}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n, f"STOP CONDITION: gate3_scale_{n}.csv has {len(rows)} rows, expected {n}"
        track_sets[n] = {row["track_id"] for row in rows}
    assert track_sets[2] <= track_sets[8] <= track_sets[32], "STOP CONDITION: TASK-05 manifests are not nested as expected"
    print(f"  manifests verified: N=2({len(track_sets[2])}) subset N=8({len(track_sets[8])}) subset N=32({len(track_sets[32])})")

    x = torch.randn(1, 48000, device=device)
    assert verify_decoder_round_trip(encodec, x), "STOP CONDITION: EnCodec decoder round-trip check failed"
    z1, z2 = encodec.encode(x), encodec.encode(x)
    assert torch.equal(z1, z2), "STOP CONDITION: EnCodec extraction is non-deterministic"
    assert z1.shape == (1, ENCODEC_CHANNELS, 150), f"STOP CONDITION: unexpected EnCodec latent shape {z1.shape}"
    print("  EnCodec decoder interface, determinism, and shape verified")


def compute_dac_oracle_metrics(waveform: torch.Tensor, dac: FrozenDacEncoder, logmel: LogMelSpectrogramEncoder, stft_loss_fn, device: torch.device) -> dict[str, float]:
    """DAC's own encode-then-decode reconstruction ceiling, computed the same way as
    `compute_oracle_audio_metrics` does for EnCodec -- `tasks/TASK-11.md`'s explicit "do not
    compare raw EnCodec waveform correlation directly with DAC without also presenting the codec
    oracle" requirement. TASK-05's own report.json never stored this number (it only stored
    translator-vs-target metrics), so it is computed fresh here from the same raw crop used to
    build the EnCodec target (`EncodecExampleData.waveform`), not recomputed from a different
    source."""
    with torch.no_grad():
        dac_target = dac.encode(waveform)
        oracle_audio = dac.decode(dac_target).squeeze(0)
    original = waveform.squeeze(0)
    original_n, oracle_n, _lag = find_best_lag_and_align(original.cpu(), oracle_audio.cpu())
    mel_a = logmel.encode(original_n.unsqueeze(0).to(device))
    mel_b = logmel.encode(oracle_n.unsqueeze(0).to(device))
    return {
        "waveform_correlation": waveform_correlation(oracle_n, original_n),
        "logmel_cosine_similarity": cosine_similarity_per_frame(mel_a, mel_b, frame_dim=1),
        "stft_distance": float(stft_loss_fn(oracle_n.unsqueeze(0).to(device), original_n.unsqueeze(0).to(device))[0].item()),
    }


def train_one_level(level: int, device: torch.device, environment: dict, wandb_mode: str, encodec: FrozenEncodecEncoder, dac: FrozenDacEncoder, dac_reports: dict) -> dict:
    print(f"\n{'=' * 70}\nLevel N={level} (EnCodec target)\n{'=' * 70}")
    manifest_path = REPO_ROOT / f"data/manifests/gate3_scale_{level}.csv"
    examples = load_examples_from_manifest_encodec(manifest_path, device, encodec, SAMPLE_RATE)
    track_ids = list(examples.keys())
    assert len(track_ids) == level, f"STOP CONDITION: loaded {len(track_ids)} examples, expected {level}"
    for fixed_track in FIXED_AUDIO_SUBSET:
        assert fixed_track in examples, f"STOP CONDITION: fixed audio-subset track {fixed_track} missing from level {level}"

    stft_loss_fn = MultiResolutionSTFTLoss(STFT_FFT_SIZES, STFT_HOP_SIZES, STFT_WIN_LENGTHS).to(device)

    print(f"  computing EnCodec oracle baseline for all {level} examples...")
    oracle_per_example = {tid: compute_oracle_audio_metrics(ex, encodec, stft_loss_fn, device) for tid, ex in examples.items()}
    oracle_agg = {
        metric: {
            "mean": float(np.mean([oracle_per_example[t][metric] for t in track_ids])),
            "median": float(np.median([oracle_per_example[t][metric] for t in track_ids])),
            "min": float(np.min([oracle_per_example[t][metric] for t in track_ids])),
            "max": float(np.max([oracle_per_example[t][metric] for t in track_ids])),
        }
        for metric in ("waveform_correlation", "logmel_cosine_similarity", "stft_distance")
    }
    print(f"  oracle (EnCodec): wav_corr_mean={oracle_agg['waveform_correlation']['mean']:.4f} logmel_cos_mean={oracle_agg['logmel_cosine_similarity']['mean']:.4f}")

    print(f"  computing DAC oracle baseline for all {level} examples...")
    logmel = LogMelSpectrogramEncoder(device)
    dac_oracle_per_example = {tid: compute_dac_oracle_metrics(ex.waveform, dac, logmel, stft_loss_fn, device) for tid, ex in examples.items()}
    dac_oracle_agg = {
        metric: {
            "mean": float(np.mean([dac_oracle_per_example[t][metric] for t in track_ids])),
            "median": float(np.median([dac_oracle_per_example[t][metric] for t in track_ids])),
            "min": float(np.min([dac_oracle_per_example[t][metric] for t in track_ids])),
            "max": float(np.max([dac_oracle_per_example[t][metric] for t in track_ids])),
        }
        for metric in ("waveform_correlation", "logmel_cosine_similarity", "stft_distance")
    }
    print(f"  oracle (DAC): wav_corr_mean={dac_oracle_agg['waveform_correlation']['mean']:.4f} logmel_cos_mean={dac_oracle_agg['logmel_cosine_similarity']['mean']:.4f}")

    config = {
        "task": "TASK-11-encodec-scale",
        "level_n": level,
        "target_codec": "encodec",
        "encodec_model_id": ENCODEC_MODEL_ID,
        "encodec_bandwidth_kbps": ENCODEC_BANDWIDTH,
        "learning_rate": LEARNING_RATE,
        "adam_betas": ADAM_BETAS,
        "cosine_loss_weight": COSINE_LOSS_WEIGHT,
        "max_steps": MAX_STEPS,
        "patience_steps": PATIENCE_STEPS,
        "val_every": VAL_EVERY,
        "seed": SEED,
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "num_examples": level,
        "wandb_version": wandb.__version__,
    }
    run = init_run(
        config=config,
        environment=environment,
        run_name=make_run_name(prefix=f"gate3_encodec_scale_{level}"),
        mode=wandb_mode,
        group="gate3-encodec-scale",
        job_type="scaling-curve",
        tags=["spectrogram", "gate3", "encodec", "scaling", f"n={level}"],
    )

    torch.manual_seed(SEED)
    model = SpectrogramToDacTranslator(input_channels=N_MELS, dac_channels=ENCODEC_CHANNELS).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()

    run_root = REPO_ROOT / "runs" / "task11_gate3_encodec_scale" / f"n{level}"
    artifact_root = REPO_ROOT / "artifacts" / "task11_gate3_encodec_scale" / f"n{level}"

    def full_latent_eval() -> dict[str, dict[str, float]]:
        per_example = {}
        with torch.no_grad():
            for track_id, ex in examples.items():
                pred = model(ex.mel)
                per_example[track_id] = evaluate_prediction(pred, ex.target, dac=None, stft_loss_fn=None, device=device, compute_audio_metrics=False)
        return per_example

    def fixed_subset_audio_eval() -> dict[str, dict[str, float]]:
        per_example = {}
        with torch.no_grad():
            for track_id in FIXED_AUDIO_SUBSET:
                ex = examples[track_id]
                pred = model(ex.mel)
                per_example[track_id] = evaluate_prediction(pred, ex.target, dac=encodec, stft_loss_fn=stft_loss_fn, device=device, compute_audio_metrics=True)
        return per_example

    best_mse = float("inf")
    best_step_since_improvement = 0
    best_wav_corr = -float("inf")
    scaling_curve: list[dict] = []
    train_start = time.perf_counter()
    step = 0
    stop_reason = "max_steps"
    while step < MAX_STEPS:
        step += 1
        track_id = track_ids[step % level]
        ex = examples[track_id]

        predicted = model(ex.mel)
        mse = mse_loss_fn(predicted, ex.target)
        cos_sim = nn.functional.cosine_similarity(predicted, ex.target, dim=-1).mean()
        total_loss = mse + COSINE_LOSS_WEIGHT * (1.0 - cos_sim)

        if not torch.isfinite(total_loss):
            raise SystemExit(f"STOP CONDITION: non-finite loss at level {level} step {step}: {total_loss.item()}")

        optimizer.zero_grad()
        total_loss.backward()
        grad_norms = three_way_grad_norms(model)
        all_grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        global_grad_norm = float(sum(g.norm(2) ** 2 for g in all_grads) ** 0.5)
        nonfinite_grads = count_nonfinite(all_grads)
        optimizer.step()

        run.log(
            {
                "train/total_loss": total_loss.item(),
                "train/mse": mse.item(),
                "train/cosine": cos_sim.item(),
                "grad/global_l2_norm": global_grad_norm,
                "grad/mel_projection_l2_norm": grad_norms["mel_projection_l2_norm"],
                "grad/translator_l2_norm": grad_norms["translator_l2_norm"],
                "grad/output_layer_l2_norm": grad_norms["output_layer_l2_norm"],
                "grad/nonfinite_count": nonfinite_grads,
            },
            step=step,
        )

        if step % VAL_EVERY == 0 or step == MAX_STEPS:
            per_example_latent = full_latent_eval()
            agg = aggregate_metrics_across_examples(per_example_latent)
            fixed_subset = fixed_subset_audio_eval()
            fixed_agg = aggregate_metrics_across_examples(fixed_subset)

            record = {
                "step": step,
                "mse_mean": agg["mse"]["mean"],
                "cosine_mean": agg["cosine"]["mean"],
                "norm_ratio_mean": agg["norm_ratio"]["mean"],
                "variance_ratio_mean": agg["variance_ratio"]["mean"],
                "waveform_correlation_to_oracle_mean": fixed_agg["waveform_correlation_to_oracle"]["mean"],
                "stft_distance_to_oracle_mean": fixed_agg["stft_distance_to_oracle"]["mean"],
            }
            scaling_curve.append(record)

            run.log(
                {
                    "collapse/norm_ratio_mean": agg["norm_ratio"]["mean"],
                    "collapse/variance_ratio_mean": agg["variance_ratio"]["mean"],
                    "audio/waveform_correlation_to_oracle": fixed_agg["waveform_correlation_to_oracle"]["mean"],
                    "audio/stft_distance_to_oracle": fixed_agg["stft_distance_to_oracle"]["mean"],
                    "eval/mse_mean": agg["mse"]["mean"],
                    "eval/cosine_mean": agg["cosine"]["mean"],
                },
                step=step,
            )

            if agg["mse"]["mean"] < best_mse - PATIENCE_MIN_DELTA:
                best_mse = agg["mse"]["mean"]
                best_step_since_improvement = step
                _save_atomic(
                    run_root / "checkpoints" / "best_by_mse.pt",
                    {"step": step, "model_state_dict": model.state_dict(), "mse_mean": best_mse},
                )

            wav_corr_now = fixed_agg["waveform_correlation_to_oracle"]["mean"]
            if wav_corr_now > best_wav_corr:
                best_wav_corr = wav_corr_now
                _save_atomic(
                    run_root / "checkpoints" / "best_by_wav_corr.pt",
                    {"step": step, "model_state_dict": model.state_dict(), "wav_corr_mean": best_wav_corr},
                )

            if step % LOG_EVERY == 0 or step == MAX_STEPS:
                print(f"  step {step}/{MAX_STEPS}: mse_mean={agg['mse']['mean']:.6f} "
                      f"cosine_mean={agg['cosine']['mean']:.4f} norm_ratio_mean={agg['norm_ratio']['mean']:.4f} "
                      f"wav_corr={wav_corr_now:.4f}")

            if step - best_step_since_improvement >= PATIENCE_STEPS:
                stop_reason = "patience_plateau"
                print(f"  stopping: no MSE improvement > {PATIENCE_MIN_DELTA} in {PATIENCE_STEPS} steps")
                break

    train_elapsed = time.perf_counter() - train_start
    print(f"  level N={level} finished: {step} steps ({stop_reason}), {train_elapsed:.2f}s")

    final_ckpt_path = run_root / "checkpoints" / "final.pt"
    _save_atomic(final_ckpt_path, {"step": step, "model_state_dict": model.state_dict()})
    reloaded = SpectrogramToDacTranslator(input_channels=N_MELS, dac_channels=ENCODEC_CHANNELS).to(device)
    reloaded.load_state_dict(torch.load(final_ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        a = model(examples[track_ids[0]].mel)
        b = reloaded(examples[track_ids[0]].mel)
    assert torch.allclose(a, b, atol=1e-6), f"STOP CONDITION: level {level} final checkpoint reload mismatch"

    def full_eval_at(state_dict) -> tuple[dict, dict, dict]:
        m = SpectrogramToDacTranslator(input_channels=N_MELS, dac_channels=ENCODEC_CHANNELS).to(device)
        m.load_state_dict(state_dict)
        m.eval()
        per_example_latent = {}
        per_example_audio = {}
        with torch.no_grad():
            for track_id, ex in examples.items():
                pred = m(ex.mel)
                per_example_latent[track_id] = evaluate_prediction(pred, ex.target, dac=None, stft_loss_fn=None, device=device, compute_audio_metrics=False)
                per_example_audio[track_id] = evaluate_prediction(pred, ex.target, dac=encodec, stft_loss_fn=stft_loss_fn, device=device, compute_audio_metrics=True)
        latent_agg = aggregate_metrics_across_examples(per_example_latent)
        audio_agg = aggregate_metrics_across_examples(per_example_audio)
        return per_example_latent, per_example_audio, {"latent": latent_agg, "audio": audio_agg}

    best_mse_state = torch.load(run_root / "checkpoints" / "best_by_mse.pt", map_location=device, weights_only=False)
    best_wav_state = torch.load(run_root / "checkpoints" / "best_by_wav_corr.pt", map_location=device, weights_only=False)
    final_state = {"model_state_dict": model.state_dict()}

    best_per_example_latent, best_per_example_audio, best_agg = full_eval_at(best_mse_state["model_state_dict"])
    _, final_per_example_audio, final_agg = full_eval_at(final_state["model_state_dict"])

    degraded = final_agg["audio"]["waveform_correlation_to_oracle"]["mean"] < best_agg["audio"]["waveform_correlation_to_oracle"]["mean"] - 0.02

    # listening package: best/median/worst by waveform_correlation_to_oracle, at the best-by-MSE checkpoint
    ranking = rank_examples_by_metric(best_per_example_audio, key="waveform_correlation_to_oracle")
    worst_id, median_id, best_id = ranking[0], ranking[len(ranking) // 2], ranking[-1]
    best_model_for_audio = SpectrogramToDacTranslator(input_channels=N_MELS, dac_channels=ENCODEC_CHANNELS).to(device)
    best_model_for_audio.load_state_dict(best_mse_state["model_state_dict"])
    best_model_for_audio.eval()
    for label, tid in [("best", best_id), ("median", median_id), ("worst", worst_id)]:
        ex = examples[tid]
        with torch.no_grad():
            pred = best_model_for_audio(ex.mel)
            pred_audio = encodec.decode(pred.transpose(1, 2)).squeeze(0).cpu()
            oracle_audio = encodec.decode(ex.target.transpose(1, 2)).squeeze(0).cpu()
        _write_wav(artifact_root / f"step{best_mse_state['step']}_{label}_{tid}.wav", pred_audio, SAMPLE_RATE)
        _write_wav(artifact_root / f"{tid}_encodec_oracle.wav", oracle_audio, SAMPLE_RATE)
        _write_wav(artifact_root / f"{tid}_original.wav", ex.waveform.squeeze(0).cpu(), SAMPLE_RATE)
    if degraded:
        for track_id in FIXED_AUDIO_SUBSET:
            ex = examples[track_id]
            fm = SpectrogramToDacTranslator(input_channels=N_MELS, dac_channels=ENCODEC_CHANNELS).to(device)
            fm.load_state_dict(final_state["model_state_dict"])
            fm.eval()
            with torch.no_grad():
                pred = fm(ex.mel)
                pred_audio = encodec.decode(pred.transpose(1, 2)).squeeze(0).cpu()
            _write_wav(artifact_root / f"step{step}_final_{track_id}.wav", pred_audio, SAMPLE_RATE)

    dac_report = dac_reports[level]
    param_diff = param_count - TASK05_PARAMETER_COUNT
    level_report = {
        "level_n": level,
        "target_codec": "encodec",
        "parameter_count": param_count,
        "parameter_count_comparison": {
            "task05_dac_parameter_count": TASK05_PARAMETER_COUNT,
            "task11_encodec_parameter_count": param_count,
            "absolute_difference": param_diff,
            "relative_difference": param_diff / TASK05_PARAMETER_COUNT,
            "note": "unavoidable dimensional change (128 vs 1024 target channels), not an intentional capacity change -- see module docstring",
        },
        "steps_run": step,
        "stop_reason": stop_reason,
        "training_time_seconds": train_elapsed,
        "best_checkpoint_step": best_mse_state["step"],
        "best_wav_corr_checkpoint_step": best_wav_state["step"],
        "final_checkpoint_step": step,
        "final_degraded_from_best": degraded,
        "scaling_curve": scaling_curve,
        "oracle_per_example": oracle_per_example,
        "oracle_aggregate": oracle_agg,
        "dac_oracle_per_example": dac_oracle_per_example,
        "dac_oracle_aggregate": dac_oracle_agg,
        "best_checkpoint": {
            "aggregate_latent_metrics": best_agg["latent"],
            "aggregate_audio_metrics": best_agg["audio"],
            "per_example_latent_metrics": best_per_example_latent,
            "per_example_audio_metrics": best_per_example_audio,
        },
        "final_checkpoint": {
            "aggregate_latent_metrics": final_agg["latent"],
            "aggregate_audio_metrics": final_agg["audio"],
            "per_example_audio_metrics": final_per_example_audio,
        },
        "listening_ranking": {"best": best_id, "median": median_id, "worst": worst_id, "ranking_metric": "waveform_correlation_to_oracle"},
        "dac_comparison": {
            "best_checkpoint_step": dac_report["best_checkpoint_step"],
            "aggregate_latent_metrics": dac_report["final_aggregate_latent_metrics"],
            "aggregate_fixed_subset_audio_metrics": dac_report["final_aggregate_fixed_subset_audio_metrics"],
        },
    }
    report_path = artifact_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(level_report, f, indent=2)
    log_artifact_file(run, report_path, f"report-encodec-n{level}", "report")
    log_artifact_file(run, run_root / "checkpoints" / "best_by_mse.pt", f"checkpoint-encodec-n{level}-best", "model")

    run.finish()
    return level_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default=",".join(str(x) for x in LEVELS), help="comma-separated list of N values to run")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    for level in levels:
        if level not in (2, 8, 32):
            raise SystemExit(f"STOP CONDITION: TASK-11 scope is N in {{2,8,32}} only, got {level}")

    try:
        device = select_device(requested="auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")
    environment = capture_environment_manifest(REPO_ROOT)

    encodec = FrozenEncodecEncoder(ENCODEC_MODEL_ID, device, bandwidth=ENCODEC_BANDWIDTH)
    n_codebooks = encodec.model.quantizer.get_num_quantizers_for_bandwidth(encodec.bandwidth)
    print(f"EnCodec: {ENCODEC_MODEL_ID}, bandwidth={ENCODEC_BANDWIDTH}kbps, n_codebooks={n_codebooks}, frame_rate={encodec.model.config.frame_rate}Hz, latent_dim={ENCODEC_CHANNELS}")
    dac = FrozenDacEncoder(DAC_MODEL_ID, device)

    gate0_static_validation(encodec, device)

    dac_reports = {}
    for level in levels:
        dac_report_path = REPO_ROOT / f"artifacts/task05_gate3_scale/n{level}/report.json"
        if not dac_report_path.exists():
            raise SystemExit(f"STOP CONDITION: {dac_report_path} missing -- cannot reuse TASK-05 DAC values")
        with open(dac_report_path) as f:
            dac_reports[level] = json.load(f)

    all_reports: dict[int, dict] = {}
    for level in levels:
        all_reports[level] = train_one_level(level, device, environment, args.wandb_mode, encodec, dac, dac_reports)

    print("\n=== EnCodec scaling summary (best-by-MSE checkpoint per level) ===")
    for level in levels:
        r = all_reports[level]
        lat = r["best_checkpoint"]["aggregate_latent_metrics"]
        aud = r["best_checkpoint"]["aggregate_audio_metrics"]
        print(f"N={level:3d}  latent_cosine={lat['cosine']['mean']:.4f}  norm_ratio={lat['norm_ratio']['mean']:.4f}  "
              f"wav_corr={aud['waveform_correlation_to_oracle']['mean']:.4f}  oracle_wav_corr={r['oracle_aggregate']['waveform_correlation']['mean']:.4f}")

    summary_path = REPO_ROOT / "artifacts/task11_gate3_encodec_scale/scaling_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"levels": levels, "full_reports": all_reports}, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
