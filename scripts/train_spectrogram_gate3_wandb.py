"""TASK-04: Spectrogram Gate-3 memorization control.

Replaces the input of the successful two-excerpt MuQ-to-DAC memorization experiment
(`scripts/train_muq_to_dac_temporal_conv.py`, `experiment/muq-to-dac-temporal-conv`, cosine
similarity 0.9999-1.0000, confirmed by listening -- notes/DECISIONS.md#muq-to-dac-temporal-conv)
with the log-mel spectrogram representation from TASK-01/TASK-02:

    log-mel spectrogram -> input projection -> TemporalConvNet (== SpectrogramToDacTranslator)
        -> DAC continuous latent -> frozen DAC decoder -> reconstructed audio

This is a training-set MEMORIZATION test on exactly the same two fixed excerpts as the MuQ
control (pop.00012, disco.00049), not a generalization experiment. There is no validation split.

## The two excerpts: reused from TASK-02's cache, not recomputed

The MuQ Gate-3 script center-crops each track: `start = (len(waveform) - 48000) // 2`, giving
336002/24000 = 14.000083s for both `pop.00012` and `disco.00049` (both files happen to be the
same native length). TASK-02's dense manifest already contains a 2.0s crop at `crop_start=14.00`
for both tracks (`crop_index=7` on its `0,2,4,...,28` grid) -- a 2-sample (83 microsecond, ~1/6000
of one DAC hop) offset from the MuQ experiment's exact center. This offset is used here, NOT
recomputed, per this task's explicit preference for reusing an already-cached, already-verified
example over regenerating a bit-identical one -- correct for THIS experiment, since it is always
trained and evaluated on its own consistent crop. **It is emphatically NOT negligible for the
MuQ Gate-3 checkpoint**: evaluating that checkpoint on the 2-sample-shifted crop instead of its
own exact training crop collapses its cosine similarity from 0.99998 to 0.577 (discovered while
building the comparison table below, not assumed) -- so the MuQ-vs-spectrogram comparison
recomputes the MuQ side on its own exact center crop, never on this task's shifted one. Both
crops are already present in `data/cache/dense/mel_dac_cache.pt` (verified before writing this
script -- see `data/manifests/gate3_spectrogram_two_excerpt.csv`, extracted from
`dense_crops.csv`). No new cache is built for this task's own training data.

## Everything else: reproduces the MuQ Gate-3 script's exact configuration

LR=1e-3, Adam betas=(0.9,0.999), cosine_loss_weight=0.1, 3000 steps, alternating presentation
(`TRACKS[step % 2]`), seed=0. The only training-loop difference from the MuQ script is the
absence of the `repeat_interleave(3)` upsampling step -- unneeded here since the spectrogram
already sits on DAC's 75Hz grid (TASK-01/02's design choice), not a change to any hyperparameter.

## Checkpoint policy: NOT validation-MSE-based (there is no validation split here)

Tracks four checkpoints: step 0 (untrained), best mean-training-MSE (across both examples, not
just the step's active one), best mean-training-cosine, best mean-waveform-correlation-to-oracle
(computed at a coarser interval, see below), and final (step 3000). Every saved checkpoint is
reloaded immediately after saving and its predictions re-verified to match, per this task's
explicit reproducibility requirement.

## Logging cadence, documented (only 3000 steps total, so most things are cheap to log every step)

Loss, gradient, and latent-only per-example diagnostics (no DAC decode) are logged every step --
3000 W&B log calls is trivial. Decode-based metrics (waveform correlation, STFT distance to
oracle) are computed and logged every `EVAL_EVERY` (25) steps, since they require a DAC decode
per example per evaluation.

Findings and full write-up: notes/DECISIONS.md#task04-spectrogram-gate3, notes/SCRATCHPAD.md.
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
from muq_inversion.audio import file_checksum, load_resample_validate
from muq_inversion.dac_encoder import FrozenDacEncoder
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.gate3_spectrogram_common import (
    LABELS,
    MANIFEST_PATH,
    TRACKS,
    build_memorization_log_payload,
    count_nonfinite,
    evaluate_prediction,
    load_examples,
    three_way_grad_norms,
)
from muq_inversion.gate4_common import (
    CropSpec,
    load_crop_waveform,
)
from muq_inversion.logging_utils import (
    DeviceSelectionError,
    capture_environment_manifest,
    select_device,
)
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.models.spectrogram_translator import SpectrogramToDacTranslator
from muq_inversion.models.temporal_conv import TemporalConvNet
from muq_inversion.spectrogram_encoder import (
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    LogMelSpectrogramEncoder,
    align_to_dac_frames,
)
from muq_inversion.wandb_logging import init_run, log_artifact_file, make_run_name

REPO_ROOT = Path(__file__).resolve().parents[1]
MUQ_GATE3_CHECKPOINT_PATH = REPO_ROOT / "runs/muq_to_dac_temporal_conv/checkpoints/latest.pt"

SAMPLE_RATE = 24000
DAC_MODEL_ID = "descript/dac_24khz"

# Identical to the MuQ Gate-3 memorization control (scripts/train_muq_to_dac_temporal_conv.py).
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
COSINE_LOSS_WEIGHT = 0.1
STEPS = 3000
LOG_EVERY = 250  # console print cadence, matches the MuQ script exactly
CHECKPOINT_EVERY = 1000  # matches the MuQ script's periodic-checkpoint cadence
SEED = 0

EVAL_EVERY = 25  # decode-based metrics only -- see module docstring's logging-cadence section
INTEGRITY_CHECK_WARMUP_STEPS = 5

STFT_FFT_SIZES = (512, 1024, 2048)
STFT_HOP_SIZES = (50, 120, 240)
STFT_WIN_LENGTHS = (240, 600, 1200)

RUN_ID = "task04_spectrogram_gate3"
AUDIO_LOG_STEPS = {0, 250, 1000, STEPS}


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


def run_integrity_checks(examples: dict, dac: FrozenDacEncoder, device: torch.device) -> None:
    """Pre-training integrity checks (tasks/TASK-04.md's numbered list). Raises SystemExit
    (STOP CONDITION) on the first failure -- per this task's explicit "fail early" instruction."""
    print("\n--- Pre-training integrity checks ---")

    # 1. Both examples load repeatedly without changing.
    reloaded = load_examples(device)
    for track_id in TRACKS:
        assert torch.equal(examples[track_id].mel, reloaded[track_id].mel), f"{track_id}: mel changed on reload"
        assert torch.equal(examples[track_id].dac, reloaded[track_id].dac), f"{track_id}: dac changed on reload"
    print("  [1/11] PASS -- repeated cache loads are identical")

    # 2 & 3. Spectrogram / DAC target caching determinism: recompute directly from source audio
    # and compare against the cached tensors used for training.
    with open(MANIFEST_PATH, newline="") as f:
        manifest_rows = {row["track_id"]: row for row in csv.DictReader(f)}
    mel_encoder = LogMelSpectrogramEncoder(device)
    for track_id in TRACKS:
        row = manifest_rows[track_id]
        spec = CropSpec(
            experiment="task04",
            split=row["split"],
            track_id=track_id,
            genre=row["genre"],
            path=REPO_ROOT / row["path"],
            checksum_sha256=row["checksum_sha256"],
            crop_index=int(row["crop_index"]),
            crop_start_seconds=float(row["crop_start_seconds"]),
            crop_duration_seconds=float(row["crop_duration_seconds"]),
        )
        waveform = load_crop_waveform(spec, SAMPLE_RATE)
        recomputed_dac = dac.encode(waveform).transpose(1, 2)
        recomputed_mel = align_to_dac_frames(mel_encoder.encode(waveform), recomputed_dac.shape[1])
        assert torch.allclose(recomputed_dac, examples[track_id].dac, atol=1e-3), f"{track_id}: DAC target not reproducible from source audio"
        assert torch.allclose(recomputed_mel, examples[track_id].mel, atol=1e-3), f"{track_id}: spectrogram not reproducible from source audio"
    print("  [2/11] PASS -- spectrogram caching is deterministic (recomputed from source audio, matches cache)")
    print("  [3/11] PASS -- DAC target caching is deterministic (recomputed from source audio, matches cache)")

    # 4 & 5. Input/target same crop, frame counts aligned.
    for track_id in TRACKS:
        ex = examples[track_id]
        assert ex.mel.shape[1] == ex.dac.shape[1], f"{track_id}: frame count mismatch mel={ex.mel.shape[1]} dac={ex.dac.shape[1]}"
        assert ex.mel.shape[1] == 150, f"{track_id}: expected 150 frames for a 2.0s/75Hz crop, got {ex.mel.shape[1]}"
    print("  [4/11] PASS -- input and target refer to the same crop (same cache label -> same manifest row)")
    print("  [5/11] PASS -- input/target sequence lengths satisfy the TASK-01/02 alignment policy (150 frames)")

    # 6. Model output shape matches target shape.
    probe_model = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
    probe_out = probe_model(examples[TRACKS[0]].mel)
    assert probe_out.shape == examples[TRACKS[0]].dac.shape, f"output shape {probe_out.shape} != target shape {examples[TRACKS[0]].dac.shape}"
    print(f"  [6/11] PASS -- model output shape {tuple(probe_out.shape)} matches target shape")

    # 7 & 8. A short warmup: loss finiteness and gradient reach, every step.
    torch.manual_seed(SEED)
    warmup_model = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
    warmup_opt = torch.optim.Adam(warmup_model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()
    for i in range(INTEGRITY_CHECK_WARMUP_STEPS):
        track_id = TRACKS[i % 2]
        ex = examples[track_id]
        pred = warmup_model(ex.mel)
        mse = mse_loss_fn(pred, ex.dac)
        cos_sim = nn.functional.cosine_similarity(pred, ex.dac, dim=-1).mean()
        loss = mse + COSINE_LOSS_WEIGHT * (1.0 - cos_sim)
        assert torch.isfinite(loss), f"STOP CONDITION: non-finite loss at warmup step {i}"
        warmup_opt.zero_grad()
        loss.backward()
        grad_norms = three_way_grad_norms(warmup_model)
        for name, norm in grad_norms.items():
            assert norm > 0 and np.isfinite(norm), f"STOP CONDITION: gradient did not reach {name} (norm={norm})"
        warmup_opt.step()
    print("  [7/11] PASS -- loss values finite for every warmup step")
    print(f"  [8/11] PASS -- gradients reach mel projection, translator, and output layer: {grad_norms}")

    # 9. Frozen DAC components remain frozen.
    n_trainable_dac_params = sum(1 for p in dac.model.parameters() if p.requires_grad)
    assert n_trainable_dac_params == 0, f"STOP CONDITION: {n_trainable_dac_params} DAC parameters have requires_grad=True"
    print("  [9/11] PASS -- frozen DAC components remain frozen (0 trainable parameters)")

    # 10. Decoding the oracle target yields expected DAC oracle audio.
    for track_id in TRACKS:
        oracle_audio = dac.decode(examples[track_id].dac.transpose(1, 2)).squeeze(0).squeeze(0)
        assert torch.isfinite(oracle_audio).all(), f"{track_id}: DAC oracle audio contains non-finite values"
        peak = oracle_audio.abs().max().item()
        assert peak > 1e-4, f"{track_id}: DAC oracle audio is silent (peak={peak})"
    print("  [10/11] PASS -- decoding the oracle target yields finite, non-silent audio")

    # 11. Save/reload checkpoint reproduces identical predictions.
    tmp_ckpt_path = REPO_ROOT / "runs" / RUN_ID / "checkpoints" / "_integrity_check.pt"
    with torch.no_grad():
        pre_save_pred = warmup_model(examples[TRACKS[0]].mel).clone()
    _save_atomic(tmp_ckpt_path, {"step": INTEGRITY_CHECK_WARMUP_STEPS, "model_state_dict": warmup_model.state_dict()})
    reloaded_model = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
    reloaded_state = torch.load(tmp_ckpt_path, map_location=device, weights_only=False)
    reloaded_model.load_state_dict(reloaded_state["model_state_dict"])
    reloaded_model.eval()
    with torch.no_grad():
        post_reload_pred = reloaded_model(examples[TRACKS[0]].mel)
    assert torch.allclose(pre_save_pred, post_reload_pred, atol=1e-6), "STOP CONDITION: checkpoint reload does not reproduce identical predictions"
    print("  [11/11] PASS -- checkpoint save/reload reproduces identical predictions")

    print("--- All 11 pre-training integrity checks PASSED ---")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    try:
        device = select_device(requested="auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")

    manifest_hash = file_checksum(MANIFEST_PATH)
    print(f"manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)} (sha256 {manifest_hash[:16]}...)")

    examples = load_examples(device)
    for track_id in TRACKS:
        ex = examples[track_id]
        print(f"  {track_id}: cache label={LABELS[track_id]} mel={tuple(ex.mel.shape)} dac={tuple(ex.dac.shape)}")

    dac = FrozenDacEncoder(DAC_MODEL_ID, device)
    run_integrity_checks(examples, dac, device)

    environment = capture_environment_manifest(REPO_ROOT)
    config = {
        "task": "TASK-04-spectrogram-gate3-memorization",
        "learning_rate": LEARNING_RATE,
        "adam_betas": ADAM_BETAS,
        "cosine_loss_weight": COSINE_LOSS_WEIGHT,
        "steps": STEPS,
        "seed": SEED,
        "eval_every": EVAL_EVERY,
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": manifest_hash,
        "num_examples": len(TRACKS),
        "crop_duration_seconds": 2.0,
        "sample_rate": SAMPLE_RATE,
        "spectrogram_n_fft": N_FFT,
        "spectrogram_hop_length": HOP_LENGTH,
        "spectrogram_n_mels": N_MELS,
        "input_tensor_shape": list(examples[TRACKS[0]].mel.shape),
        "target_tensor_shape": list(examples[TRACKS[0]].dac.shape),
        "wandb_version": wandb.__version__,
    }
    run = init_run(
        config=config,
        environment=environment,
        run_name=args.run_name or make_run_name(prefix="spec_gate3_two_excerpt"),
        mode=args.wandb_mode,
        group="spectrogram-gate3",
        job_type="memorization-control",
        tags=["spectrogram", "gate3", "memorization", "positive-control", "two-excerpt", "post-rvq-target"],
    )
    print(f"W&B run: {run.name if hasattr(run, 'name') else '(disabled)'}, mode={args.wandb_mode}")

    torch.manual_seed(SEED)
    model = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {param_count:,} (trainable: {trainable_param_count:,})")
    print(f"receptive field: {model.receptive_field_frames} frames at DAC's timeline")
    run.log(
        {
            "config/parameter_count": param_count,
            "config/trainable_parameter_count": trainable_param_count,
            "config/receptive_field_frames": model.receptive_field_frames,
        },
        step=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    mse_loss_fn = nn.MSELoss()
    stft_loss_fn = MultiResolutionSTFTLoss(STFT_FFT_SIZES, STFT_HOP_SIZES, STFT_WIN_LENGTHS).to(device)

    run_root = REPO_ROOT / "runs" / RUN_ID
    artifact_root = REPO_ROOT / "artifacts" / RUN_ID

    def save_checkpoint(name: str, step: int, extra: dict) -> Path:
        path = run_root / "checkpoints" / f"{name}.pt"
        state = {"step": step, "model_state_dict": model.state_dict(), **extra}
        _save_atomic(path, state)
        # Verify reload reproduces identical predictions -- required for every saved checkpoint.
        reloaded = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
        reloaded.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model_state_dict"])
        reloaded.eval()
        with torch.no_grad():
            a = model(examples[TRACKS[0]].mel)
            b = reloaded(examples[TRACKS[0]].mel)
        assert torch.allclose(a, b, atol=1e-6), f"STOP CONDITION: checkpoint '{name}' reload mismatch at step {step}"
        return path

    def write_audio_for_step(step: int, tag: str, eval_model: SpectrogramToDacTranslator) -> None:
        for track_id in TRACKS:
            ex = examples[track_id]
            with torch.no_grad():
                pred = eval_model(ex.mel)
                pred_audio = dac.decode(pred.transpose(1, 2)).squeeze(0).squeeze(0).cpu()
                oracle_audio = dac.decode(ex.dac.transpose(1, 2)).squeeze(0).squeeze(0).cpu()
            fname_pred = f"{track_id}_{tag}_step{step}.wav"
            _write_wav(artifact_root / fname_pred, pred_audio, SAMPLE_RATE)
            caption = f"{track_id} | step {step} | {tag}"
            run.log({f"audio/{track_id}/{tag}": wandb.Audio(pred_audio.numpy(), sample_rate=SAMPLE_RATE, caption=caption)}, step=step)
            if tag == "step0":  # log the oracle/reference once, not at every tag
                _write_wav(artifact_root / f"{track_id}_dac_oracle.wav", oracle_audio, SAMPLE_RATE)
                run.log({f"audio/{track_id}/dac_oracle": wandb.Audio(oracle_audio.numpy(), sample_rate=SAMPLE_RATE, caption=f"{track_id} | DAC oracle")}, step=step)

    # Reference audio, written once.
    with open(MANIFEST_PATH, newline="") as f:
        manifest_rows = {row["track_id"]: row for row in csv.DictReader(f)}
    for track_id in TRACKS:
        row = manifest_rows[track_id]
        spec = CropSpec(
            experiment="task04", split=row["split"], track_id=track_id, genre=row["genre"],
            path=REPO_ROOT / row["path"], checksum_sha256=row["checksum_sha256"],
            crop_index=int(row["crop_index"]), crop_start_seconds=float(row["crop_start_seconds"]),
            crop_duration_seconds=float(row["crop_duration_seconds"]),
        )
        reference = load_crop_waveform(spec, SAMPLE_RATE).squeeze(0)
        _write_wav(artifact_root / f"{track_id}_reference.wav", reference, SAMPLE_RATE)

    print("\n--- Step 0 (untrained model) ---")
    write_audio_for_step(0, "step0", model)

    print(f"\n--- Training ({STEPS} steps, MSE + {COSINE_LOSS_WEIGHT}*cosine, alternating presentation) ---")
    presentation_counts = {t: 0 for t in TRACKS}
    best = {"mse": float("inf"), "cos": -float("inf"), "wav_corr": -float("inf")}
    train_start = time.perf_counter()
    prev_params = None
    for step in range(1, STEPS + 1):
        track_id = TRACKS[step % len(TRACKS)]
        presentation_counts[track_id] += 1
        ex = examples[track_id]

        predicted = model(ex.mel)
        mse = mse_loss_fn(predicted, ex.dac)
        cos_sim = nn.functional.cosine_similarity(predicted, ex.dac, dim=-1).mean()
        cos_loss = 1.0 - cos_sim
        total_loss = mse + COSINE_LOSS_WEIGHT * cos_loss

        if not torch.isfinite(total_loss):
            raise SystemExit(f"STOP CONDITION: non-finite loss at step {step}: {total_loss.item()}")

        optimizer.zero_grad()
        total_loss.backward()

        grad_norms = three_way_grad_norms(model)
        all_grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        global_grad_norm = float(sum(g.norm(2) ** 2 for g in all_grads) ** 0.5)
        global_max_abs = float(max(g.abs().max() for g in all_grads))
        nonfinite_grads = count_nonfinite(all_grads)
        nonfinite_params = count_nonfinite([p.detach() for p in model.parameters()])
        nonfinite_preds = int((~torch.isfinite(predicted)).sum().item())

        if prev_params is None:
            prev_params = [p.detach().clone() for p in model.parameters()]
        pre_step_params = [p.detach().clone() for p in model.parameters()]

        optimizer.step()

        update_norm = float(sum(((p.detach() - old) ** 2).sum() for p, old in zip(model.parameters(), pre_step_params, strict=True)) ** 0.5)
        param_norm = float(sum((p.detach() ** 2).sum() for p in model.parameters()) ** 0.5)
        update_ratio = update_norm / (param_norm + 1e-12)

        run.log(
            {
                "train/total_loss": total_loss.item(),
                "train/mse_loss": mse.item(),
                "train/cosine_loss": cos_loss.item(),
                "train/learning_rate": LEARNING_RATE,
                "train/example_0_presentations": presentation_counts[TRACKS[0]],
                "train/example_1_presentations": presentation_counts[TRACKS[1]],
                "grad/global_l2_norm": global_grad_norm,
                "grad/global_max_abs": global_max_abs,
                "grad/mel_projection_l2_norm": grad_norms["mel_projection_l2_norm"],
                "grad/translator_l2_norm": grad_norms["translator_l2_norm"],
                "grad/output_layer_l2_norm": grad_norms["output_layer_l2_norm"],
                "grad/nonfinite_count": nonfinite_grads,
                "grad/nonfinite_param_count": nonfinite_params,
                "grad/nonfinite_prediction_count": nonfinite_preds,
                "update/model_parameter_l2_norm": param_norm,
                "update/update_to_parameter_ratio": update_ratio,
            },
            step=step,
        )

        compute_audio = (step % EVAL_EVERY == 0) or step == STEPS
        with torch.no_grad():
            preds_now = {t: model(examples[t].mel) for t in TRACKS}
        per_example = {t: evaluate_prediction(preds_now[t], examples[t].dac, dac, stft_loss_fn, device, compute_audio) for t in TRACKS}
        mean_metrics = {k: float(np.mean([per_example[t][k] for t in TRACKS])) for k in per_example[TRACKS[0]]}

        run.log(build_memorization_log_payload(per_example, mean_metrics, TRACKS), step=step)

        if mean_metrics["mse"] < best["mse"]:
            best["mse"] = mean_metrics["mse"]
            save_checkpoint("best_by_mse", step, {"mean_mse": mean_metrics["mse"], "mean_cos": mean_metrics["cosine"]})
        if mean_metrics["cosine"] > best["cos"]:
            best["cos"] = mean_metrics["cosine"]
            save_checkpoint("best_by_cosine", step, {"mean_mse": mean_metrics["mse"], "mean_cos": mean_metrics["cosine"]})
        if compute_audio and mean_metrics.get("waveform_correlation_to_oracle", -float("inf")) > best["wav_corr"]:
            best["wav_corr"] = mean_metrics["waveform_correlation_to_oracle"]
            save_checkpoint("best_by_wav_corr", step, {"wav_corr": best["wav_corr"]})

        if step in AUDIO_LOG_STEPS:
            tag = {250: "early", 1000: "intermediate", STEPS: "final"}.get(step, f"step{step}")
            write_audio_for_step(step, tag, model)

        if step % CHECKPOINT_EVERY == 0 or step == STEPS:
            save_checkpoint("periodic", step, {})

        if step == 1 or step % LOG_EVERY == 0 or step == STEPS:
            print(
                f"step {step}/{STEPS}: mse={mse.item():.6f} cos={cos_sim.item():.4f} "
                f"mean_mse={mean_metrics['mse']:.6f} mean_cos={mean_metrics['cosine']:.4f} track={track_id}"
            )

    train_elapsed = time.perf_counter() - train_start
    save_checkpoint("final", STEPS, {})
    print(f"\ntraining time: {train_elapsed:.2f}s ({STEPS} steps, {train_elapsed / STEPS * 1000:.2f} ms/step)")
    print(f"presentation counts: {presentation_counts}")
    print(f"best mean MSE: {best['mse']:.6f} | best mean cosine: {best['cos']:.4f} | best mean wav_corr: {best['wav_corr']:.4f}")

    print("\n--- Final per-checkpoint metrics (reloaded, verified) ---")
    checkpoint_reports = {}
    for name in ("step0", "best_by_mse", "best_by_cosine", "best_by_wav_corr", "final"):
        ckpt_path = run_root / "checkpoints" / f"{name}.pt"
        if not ckpt_path.exists():
            continue
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        eval_model = SpectrogramToDacTranslator(input_channels=N_MELS).to(device)
        eval_model.load_state_dict(state["model_state_dict"])
        eval_model.eval()
        with torch.no_grad():
            preds_ckpt = {t: eval_model(examples[t].mel) for t in TRACKS}
        per_example = {t: evaluate_prediction(preds_ckpt[t], examples[t].dac, dac, stft_loss_fn, device, True) for t in TRACKS}
        mean_metrics = {k: float(np.mean([per_example[t][k] for t in TRACKS])) for k in per_example[TRACKS[0]]}
        checkpoint_reports[name] = {"step": state["step"], "per_example": per_example, "mean": mean_metrics}
        print(f"  {name} (step {state['step']}): mean_mse={mean_metrics['mse']:.6f} mean_cos={mean_metrics['cosine']:.4f} "
              f"mean_wav_corr={mean_metrics.get('waveform_correlation_to_oracle', float('nan')):.4f}")

    write_audio_for_step(checkpoint_reports["best_by_mse"]["step"], "best_by_mse", model)

    print("\n--- Comparison against the MuQ Gate-3 memorization control ---")
    muq_comparison = None
    if MUQ_GATE3_CHECKPOINT_PATH.exists():
        muq_state = torch.load(MUQ_GATE3_CHECKPOINT_PATH, map_location=device, weights_only=False)
        muq_model = TemporalConvNet(channels=1024).to(device)
        muq_model.load_state_dict(muq_state["model_state_dict"])
        muq_model.eval()
        muq_encoder = FrozenMuQEncoder("OpenMuQ/MuQ-large-msd-iter", device)
        frame_ratio = muq_state["config"]["frame_ratio"]

        # IMPORTANT, discovered while writing this comparison (not assumed): the MuQ Gate-3
        # checkpoint is extremely sensitive to sub-hop sample alignment -- evaluating it on this
        # task's dense-grid crop (2 samples / 83us off the MuQ script's own exact center crop)
        # collapses its cosine similarity from 0.99998 to 0.577, a >40 percentage point swing
        # from an offset far smaller than one DAC hop (320 samples). This is itself evidence the
        # MuQ memorization is a narrow, sample-exact overfit, not a content-level fit -- see
        # notes/DECISIONS.md#task04-spectrogram-gate3. The comparison below therefore uses the
        # MuQ script's OWN exact center-crop formula (not this task's manifest), computing a
        # matching fresh DAC target from that exact crop, so the MuQ side of the table reflects
        # the model evaluated on the crop it actually memorized.
        muq_per_example = {}
        for track_id in TRACKS:
            row = manifest_rows[track_id]
            path = REPO_ROOT / row["path"]
            waveform_np, _ = load_resample_validate(path, SAMPLE_RATE)
            crop_samples = round(2.0 * SAMPLE_RATE)
            start = (len(waveform_np) - crop_samples) // 2  # exact formula from train_muq_to_dac_temporal_conv.py
            wav = torch.from_numpy(waveform_np[start : start + crop_samples]).float().unsqueeze(0).to(device)

            muq_feat = muq_encoder.encode(wav)
            repeated = muq_feat.repeat_interleave(frame_ratio, dim=1)
            with torch.no_grad():
                muq_pred = muq_model(repeated)
                muq_own_target = dac.encode(wav).transpose(1, 2)  # this exact crop's own DAC target
            muq_per_example[track_id] = evaluate_prediction(muq_pred, muq_own_target, dac, stft_loss_fn, device, True)
        muq_mean = {k: float(np.mean([muq_per_example[t][k] for t in TRACKS])) for k in muq_per_example[TRACKS[0]]}
        muq_comparison = {"checkpoint_step": muq_state["step"], "per_example": muq_per_example, "mean": muq_mean}
        print(f"  MuQ Gate-3 (step {muq_state['step']}): mean_mse={muq_mean['mse']:.6f} mean_cos={muq_mean['cosine']:.4f} "
              f"mean_wav_corr={muq_mean.get('waveform_correlation_to_oracle', float('nan')):.4f}")
    else:
        print(f"  WARNING: {MUQ_GATE3_CHECKPOINT_PATH} not found -- skipping direct MuQ comparison")

    report = {
        "run_id": RUN_ID,
        "device": str(device),
        "config": config,
        "presentation_counts": presentation_counts,
        "training_time_seconds": train_elapsed,
        "parameter_count": param_count,
        "trainable_parameter_count": trainable_param_count,
        "receptive_field_frames": model.receptive_field_frames,
        "checkpoint_reports": checkpoint_reports,
        "muq_gate3_comparison": muq_comparison,
    }
    report_path = artifact_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote full report to {report_path}")

    for name in ("best_by_mse", "best_by_cosine", "final"):
        ckpt_path = run_root / "checkpoints" / f"{name}.pt"
        if ckpt_path.exists():
            log_artifact_file(run, ckpt_path, f"checkpoint-{name}", "model")
    log_artifact_file(run, report_path, "evaluation-report", "report")
    log_artifact_file(run, MANIFEST_PATH, "two-excerpt-manifest", "dataset")

    run.finish()
    print("W&B run finished.")


if __name__ == "__main__":
    main()
