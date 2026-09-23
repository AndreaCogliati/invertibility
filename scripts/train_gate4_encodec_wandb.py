"""TASK-13: Gate 4 revalidation with the merged (post-`experiment/muq-layer-sweep`) architecture.

Repeats CLAUDE.md's Gate 4 short-pilot protocol -- checkpoint/resume equivalence, then
conditioning-control comparison -- on the pipeline actually on `main` after the merge:

    frozen MuQ `last_hidden_state` (25Hz, 1024ch)
        -> repeat_interleave to EnCodec's 75Hz frame rate (measured, not assumed)
        -> SpectrogramToDacTranslator (unmodified; here input_channels=1024, dac_channels=128)
        -> frozen EnCodec continuous latent -> frozen EnCodec decoder -> reconstructed audio

The previous Gate 4 result (`notes/DECISIONS.md#gate4-shuffled-time-finding`, the older
HiFi-GAN-style waveform-output pipeline) found `shuffled_across` and `constant_zero` clearly
underperformed correct conditioning, but `shuffled_time` (frames permuted in time) did not --
Gate 4 was reported not fully passed. This script re-runs the identical four-condition comparison
-- `correct`, `shuffled_across`, `shuffled_time`, `constant_zero` -- against a single trained
checkpoint (this task's own explicit evaluation protocol, distinct from the older pilot.py's
separate-model-per-control approach) on the current EnCodec-target architecture.

Manifest: `data/manifests/gate4_crops.csv`, Experiment B (8 train tracks / 2 crops each = 16 train
crops, 2 val tracks, 2 test tracks, all track-disjoint) -- the existing Gate 4 manifest closest in
scale and structure to CLAUDE.md's "small deterministic subset" Gate 4 description, reused
unchanged rather than building a new one.

Findings and full write-up: notes/DECISIONS.md#task13-gate4-encodec-revalidation, notes/SCRATCHPAD.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

# Must be set before CUDA is initialized (i.e. before `import torch` touches the GPU) --
# required for `torch.use_deterministic_algorithms(True)` to cover cuBLAS ops below. Harmless
# on MPS/CPU-only machines, where it is simply never read.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import soundfile as sf
import torch
from torch import nn

import wandb
from muq_inversion.encodec_encoder import FrozenEncodecEncoder
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.gate3_spectrogram_common import (
    aggregate_metrics_across_examples,
    count_nonfinite,
    evaluate_prediction,
    rank_examples_by_metric,
    three_way_grad_norms,
)
from muq_inversion.gate4_encodec_common import (
    CONDITIONING_MODES,
    Gate4Example,
    build_conditioning,
    load_gate4_examples,
)
from muq_inversion.logging_utils import (
    DeviceSelectionError,
    capture_environment_manifest,
    select_device,
)
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.models.spectrogram_translator import SpectrogramToDacTranslator
from muq_inversion.wandb_logging import init_run, log_artifact_file, make_run_name

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data/manifests/gate4_crops.csv"
EXPERIMENT = "B"
SAMPLE_RATE = 24000
MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
ENCODEC_MODEL_ID = "facebook/encodec_24khz"
ENCODEC_BANDWIDTH = 24.0
MUQ_CHANNELS = 1024
ENCODEC_CHANNELS = 128

# Unchanged from every prior MuQ/spectrogram-to-EnCodec experiment in this project (TASK-05/11/12).
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
COSINE_LOSS_WEIGHT = 0.1
SEED = 0

STFT_FFT_SIZES = (512, 1024, 2048)
STFT_HOP_SIZES = (50, 120, 240)
STFT_WIN_LENGTHS = (240, 600, 1200)

STEPS = 20000  # matches TASK-11's own N=32 EnCodec budget; this run has fewer (16) train examples
VAL_EVERY = 500
LOG_EVERY = 200
RESUME_TEST_AT = STEPS // 2
RESUME_TOLERANCE = 0.15  # relative; MPS is not bit-exact across invocations (see gate4-mps-resume-tolerance)

RUN_ID = "task13_gate4_encodec_revalidation"


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _rng_snapshot() -> dict:
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_mps": mps_state,
        "torch_cuda": cuda_state,
    }


def _rng_restore(snapshot: dict) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])
    if snapshot.get("torch_mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(snapshot["torch_mps"])
    if snapshot.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def _enable_cuda_determinism(device: torch.device) -> None:
    """`tasks/TASK-13.md` requires a checkpoint/resume equivalence *check*, not just a checkpoint
    write -- on a first BlueHive (CUDA) run without this, the check correctly FAILED (max relative
    diff 8.4 pre-checkpoint, 153.8 post-resume) because cuDNN convolution/cuBLAS kernels are not
    bit-reproducible across separate process invocations by default, unlike this project's MPS
    runs (which measured an exact 0.0000 match twice). `warn_only=True`: falls back to a
    best-effort deterministic kernel with a warning rather than raising, for any op in this
    pipeline that has no deterministic CUDA implementation, instead of crashing the whole run."""
    if device.type != "cuda":
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _make_model_and_optimizer(device: torch.device, seed: int) -> tuple[SpectrogramToDacTranslator, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    model = SpectrogramToDacTranslator(input_channels=MUQ_CHANNELS, dac_channels=ENCODEC_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    return model, optimizer


def _train_step(
    model: SpectrogramToDacTranslator,
    optimizer: torch.optim.Optimizer,
    mse_loss_fn: nn.Module,
    example: Gate4Example,
    frame_ratio: int,
) -> dict[str, float]:
    repeated = example.muq.repeat_interleave(frame_ratio, dim=1)
    predicted = model(repeated)
    mse = mse_loss_fn(predicted, example.target)
    cos_sim = nn.functional.cosine_similarity(predicted, example.target, dim=-1).mean()
    total_loss = mse + COSINE_LOSS_WEIGHT * (1.0 - cos_sim)

    if not torch.isfinite(total_loss):
        raise SystemExit(f"STOP CONDITION: non-finite loss: {total_loss.item()}")

    optimizer.zero_grad()
    total_loss.backward()
    grad_norms = three_way_grad_norms(model)
    all_grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    global_grad_norm = float(sum(g.norm(2) ** 2 for g in all_grads) ** 0.5)
    nonfinite_grads = count_nonfinite(all_grads)
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "mse": mse.item(),
        "cosine": cos_sim.item(),
        "global_grad_norm": global_grad_norm,
        "mel_projection_l2_norm": grad_norms["mel_projection_l2_norm"],
        "translator_l2_norm": grad_norms["translator_l2_norm"],
        "output_layer_l2_norm": grad_norms["output_layer_l2_norm"],
        "nonfinite_grads": nonfinite_grads,
    }


def compute_val_mse(
    model: SpectrogramToDacTranslator, examples: list[Gate4Example], frame_ratio: int, mse_loss_fn: nn.Module
) -> float:
    if not examples:
        return float("nan")
    total = 0.0
    with torch.no_grad():
        for ex in examples:
            repeated = ex.muq.repeat_interleave(frame_ratio, dim=1)
            predicted = model(repeated)
            total += mse_loss_fn(predicted, ex.target).item()
    return total / len(examples)


def run_training(
    train_examples: list[Gate4Example],
    val_examples: list[Gate4Example],
    frame_ratio: int,
    device: torch.device,
    seed: int,
    steps: int,
    log_fn,
    val_log_fn=None,
) -> tuple[list[dict[str, float]], SpectrogramToDacTranslator, torch.optim.Optimizer, random.Random]:
    """Trains `steps` steps from scratch (seeded), returns the per-step training-metric history,
    the final model/optimizer, and the crop-selection RNG (for checkpointing/resume)."""
    model, optimizer = _make_model_and_optimizer(device, seed)
    mse_loss_fn = nn.MSELoss()
    rng = random.Random(seed)
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        example = train_examples[rng.randrange(len(train_examples))]
        metrics = _train_step(model, optimizer, mse_loss_fn, example, frame_ratio)
        history.append(metrics)
        if log_fn is not None:
            log_fn(step, metrics)
        if step % VAL_EVERY == 0 or step == steps:
            val_mse = compute_val_mse(model, val_examples, frame_ratio, mse_loss_fn)
            if val_log_fn is not None:
                val_log_fn(step, val_mse)
        if step % LOG_EVERY == 0 or step == steps:
            val_mse = compute_val_mse(model, val_examples, frame_ratio, mse_loss_fn)
            print(f"  step {step}/{steps}: loss={metrics['total_loss']:.4f} mse={metrics['mse']:.4f} "
                  f"cos={metrics['cosine']:.4f} val_mse={val_mse:.4f}")
    return history, model, optimizer, rng


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--steps", type=int, default=STEPS, help="override the default 20000-step budget (smoke tests)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = args.steps
    resume_test_at = steps // 2

    try:
        device = select_device(requested="auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")
    _enable_cuda_determinism(device)
    environment = capture_environment_manifest(REPO_ROOT)

    muq = FrozenMuQEncoder(MUQ_MODEL_ID, device)
    encodec = FrozenEncodecEncoder(ENCODEC_MODEL_ID, device, bandwidth=ENCODEC_BANDWIDTH)

    print(f"\n--- Loading Gate 4 manifest (experiment {EXPERIMENT}) and precomputing MuQ/EnCodec latents ---")
    examples, frame_ratio = load_gate4_examples(MANIFEST_PATH, EXPERIMENT, muq, encodec, SAMPLE_RATE, device)
    print(f"loaded {len(examples)} crops, frame_ratio={frame_ratio} (MuQ 25Hz -> EnCodec 75Hz)")

    by_split: dict[str, list[Gate4Example]] = {"train": [], "val": [], "test": []}
    for ex in examples.values():
        by_split[ex.spec.split].append(ex)
    for split, split_examples in by_split.items():
        tracks = sorted({e.spec.track_id for e in split_examples})
        print(f"  {split}: {len(split_examples)} crops, tracks={tracks}")

    # Re-verify track-split disjointness at run time (not just at manifest-build time).
    track_splits: dict[str, set[str]] = {}
    for ex in examples.values():
        track_splits.setdefault(ex.spec.track_id, set()).add(ex.spec.split)
    for track_id, splits_seen in track_splits.items():
        assert len(splits_seen) == 1, f"STOP CONDITION: {track_id} spans splits {splits_seen}"

    all_examples = list(examples.values())
    param_count = None  # set after first model construction below

    manifest_hash = _file_hash(MANIFEST_PATH)
    uv_lock_hash = _file_hash(REPO_ROOT / "uv.lock")

    config = {
        "task": "TASK-13",
        "gate": "Gate 4 revalidation",
        "git_commit": environment.get("git_commit"),
        "git_dirty": environment.get("git_dirty"),
        "muq_model_id": MUQ_MODEL_ID,
        "encodec_model_id": ENCODEC_MODEL_ID,
        "encodec_bandwidth_kbps": ENCODEC_BANDWIDTH,
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_experiment": EXPERIMENT,
        "manifest_hash": manifest_hash,
        "uv_lock_hash": uv_lock_hash,
        "seed": SEED,
        "device": str(device),
        "dtype": "float32",
        "frame_ratio": frame_ratio,
        "num_train_examples": len(by_split["train"]),
        "num_val_examples": len(by_split["val"]),
        "num_test_examples": len(by_split["test"]),
        "learning_rate": LEARNING_RATE,
        "adam_betas": ADAM_BETAS,
        "cosine_loss_weight": COSINE_LOSS_WEIGHT,
        "loss_weights": {"mse": 1.0, "cosine": COSINE_LOSS_WEIGHT},
        "batch_size": 1,
        "steps": steps,
        "val_every": VAL_EVERY,
        "resume_test_at": resume_test_at,
        "resume_tolerance": RESUME_TOLERANCE,
        "checkpoint_selection_metric": "val_mse",
        "conditioning_modes": list(CONDITIONING_MODES),
        "wandb_version": wandb.__version__,
    }
    run = init_run(
        config=config,
        environment=environment,
        run_name=make_run_name(prefix="task13_gate4_encodec"),
        mode=args.wandb_mode,
        group="task13-gate4-encodec-revalidation",
        job_type="gate4-revalidation",
        tags=["gate4", "encodec", "task13", "merged-architecture"],
    )

    run_root = REPO_ROOT / "runs" / RUN_ID
    artifact_root = REPO_ROOT / "artifacts" / RUN_ID
    checkpoint_dir = run_root / "checkpoints"

    # ---- Checkpoint/resume equivalence check ----
    print(f"\n--- Checkpoint/resume equivalence check (resume at step {resume_test_at}/{steps}) ---")

    def log_uninterrupted(step: int, metrics: dict[str, float]) -> None:
        run.log(
            {
                "train/total_loss": metrics["total_loss"],
                "train/mse": metrics["mse"],
                "train/cosine": metrics["cosine"],
                "grad/global_l2_norm": metrics["global_grad_norm"],
                "grad/mel_projection_l2_norm": metrics["mel_projection_l2_norm"],
                "grad/translator_l2_norm": metrics["translator_l2_norm"],
                "grad/output_layer_l2_norm": metrics["output_layer_l2_norm"],
                "grad/nonfinite_count": metrics["nonfinite_grads"],
            },
            step=step,
        )

    best_val_state = {"mse": float("inf"), "step": 0}

    def log_val(step: int, val_mse: float) -> None:
        if val_mse < best_val_state["mse"]:
            best_val_state["mse"] = val_mse
            best_val_state["step"] = step
        run.log(
            {"val/mse": val_mse, "val/best_mse_so_far": best_val_state["mse"], "train/learning_rate": LEARNING_RATE},
            step=step,
        )

    uninterrupted_history, _model_a, _opt_a, _rng_a = run_training(
        by_split["train"], by_split["val"], frame_ratio, device, SEED, steps,
        log_fn=log_uninterrupted, val_log_fn=log_val,
    )

    print("  running pre-checkpoint half (fresh model, same seed) ...")
    model_b, opt_b = _make_model_and_optimizer(device, SEED)
    mse_loss_fn = nn.MSELoss()
    rng_b = random.Random(SEED)
    first_half_history: list[dict[str, float]] = []
    for step in range(1, resume_test_at + 1):
        example = by_split["train"][rng_b.randrange(len(by_split["train"]))]
        first_half_history.append(_train_step(model_b, opt_b, mse_loss_fn, example, frame_ratio))

    max_predeterminism_diff = max(
        abs(a["total_loss"] - b["total_loss"]) / max(abs(a["total_loss"]), 1e-6)
        for a, b in zip(uninterrupted_history[:resume_test_at], first_half_history, strict=True)
    )
    determinism_ok = max_predeterminism_diff <= RESUME_TOLERANCE
    print(f"  pre-checkpoint determinism match: {determinism_ok} (max relative diff {max_predeterminism_diff:.4f})")

    checkpoint_path = checkpoint_dir / "resume_test_checkpoint.pt"
    _save_atomic(
        checkpoint_path,
        {
            "step": resume_test_at,
            "model_state_dict": model_b.state_dict(),
            "optimizer_state_dict": opt_b.state_dict(),
            "rng_state": _rng_snapshot(),
            "crop_rng_state": rng_b.getstate(),
            "resolved_config": config,
            "manifest_hash": manifest_hash,
            "uv_lock_hash": uv_lock_hash,
            "git_commit": environment.get("git_commit"),
            "git_dirty": environment.get("git_dirty"),
        },
    )
    # map_location="cpu": the checkpoint's RNG state tensors must stay CPU ByteTensors for
    # torch.set_rng_state; the model/optimizer state_dicts are moved to `device` separately below
    # via load_state_dict (which relocates parameters onto the target module automatically).
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "step", "resolved_config", "manifest_hash", "uv_lock_hash"):
        assert key in reloaded, f"STOP CONDITION: checkpoint missing required key {key!r}"

    # Fresh model/optimizer from a *different* seed -- proves the checkpoint load, not
    # initialization, determines the resumed trajectory.
    model_c, opt_c = _make_model_and_optimizer(device, SEED + 1)
    model_c.load_state_dict(reloaded["model_state_dict"])
    opt_c.load_state_dict(reloaded["optimizer_state_dict"])
    _rng_restore(reloaded["rng_state"])
    rng_c = random.Random()
    rng_c.setstate(reloaded["crop_rng_state"])

    print("  running resumed second half ...")
    resumed_second_half_history: list[dict[str, float]] = []
    for step in range(resume_test_at + 1, steps + 1):
        example = by_split["train"][rng_c.randrange(len(by_split["train"]))]
        resumed_second_half_history.append(_train_step(model_c, opt_c, mse_loss_fn, example, frame_ratio))

    uninterrupted_second_half = uninterrupted_history[resume_test_at:]
    max_resume_diff = max(
        abs(a["total_loss"] - b["total_loss"]) / max(abs(a["total_loss"]), 1e-6)
        for a, b in zip(uninterrupted_second_half, resumed_second_half_history, strict=True)
    )
    resume_ok = determinism_ok and max_resume_diff <= RESUME_TOLERANCE
    print(f"  max relative loss diff (resumed vs uninterrupted, 2nd half): {max_resume_diff:.4f} "
          f"(tolerance {RESUME_TOLERANCE})")
    print(f"  checkpoint/resume equivalence: {'PASSED' if resume_ok else 'FAILED'}")

    run.log({
        "resume_check/predeterminism_max_rel_diff": max_predeterminism_diff,
        "resume_check/resume_max_rel_diff": max_resume_diff,
        "resume_check/passed": float(resume_ok),
        "resume_check/tolerance": RESUME_TOLERANCE,
    })

    # ---- Main run: the model used for the conditioning-control comparison ----
    # `uninterrupted_history`'s model (`_model_a`) already trained the full budget with correct
    # conditioning above -- reused directly as "the trained checkpoint" this task's control
    # evaluation protocol requires, rather than training a fourth redundant copy.
    model = _model_a
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nparameters: {param_count:,}  receptive_field: {model.receptive_field_frames} frames")

    final_val_mse = compute_val_mse(model, by_split["val"], frame_ratio, mse_loss_fn)
    run.log({"val/final_mse": final_val_mse}, step=steps)

    final_ckpt_path = checkpoint_dir / "final.pt"
    _save_atomic(
        final_ckpt_path,
        {
            "step": steps,
            "model_state_dict": model.state_dict(),
            "final_val_mse": final_val_mse,
            "resolved_config": config,
            "manifest_hash": manifest_hash,
            "uv_lock_hash": uv_lock_hash,
            "git_commit": environment.get("git_commit"),
            "git_dirty": environment.get("git_dirty"),
        },
    )
    reloaded_final = SpectrogramToDacTranslator(input_channels=MUQ_CHANNELS, dac_channels=ENCODEC_CHANNELS).to(device)
    reloaded_final.load_state_dict(torch.load(final_ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    reloaded_final.eval()
    with torch.no_grad():
        probe_ex = all_examples[0]
        a = model(probe_ex.muq.repeat_interleave(frame_ratio, dim=1))
        b = reloaded_final(probe_ex.muq.repeat_interleave(frame_ratio, dim=1))
    assert torch.allclose(a, b, atol=1e-6), "STOP CONDITION: final checkpoint reload mismatch"

    # ---- Conditioning-control evaluation on the single trained checkpoint ----
    print("\n--- Conditioning-control evaluation ---")
    stft_loss_fn = MultiResolutionSTFTLoss(STFT_FFT_SIZES, STFT_HOP_SIZES, STFT_WIN_LENGTHS).to(device)
    model.eval()

    control_rng = random.Random(SEED)
    control_report: dict[str, dict] = {}
    for mode in CONDITIONING_MODES:
        per_example: dict[str, dict[str, float]] = {}
        for label, ex in examples.items():
            with torch.no_grad():
                conditioned = build_conditioning(mode, ex, all_examples, frame_ratio, control_rng)
                pred = model(conditioned)
            per_example[label] = evaluate_prediction(
                pred, ex.target, dac=encodec, stft_loss_fn=stft_loss_fn, device=device, compute_audio_metrics=True
            )
        control_report[mode] = {
            "per_example": per_example,
            "aggregate": aggregate_metrics_across_examples(per_example),
            "aggregate_by_split": {
                split: aggregate_metrics_across_examples(
                    {lbl: m for lbl, m in per_example.items() if examples[lbl].spec.split == split}
                )
                for split in ("train", "val", "test")
            },
        }
        agg = control_report[mode]["aggregate"]
        print(f"  {mode}: wav_corr_to_oracle mean={agg['waveform_correlation_to_oracle']['mean']:.4f} "
              f"variance_ratio mean={agg['variance_ratio']['mean']:.4f} cosine mean={agg['cosine']['mean']:.4f}")

    for mode in CONDITIONING_MODES:
        agg = control_report[mode]["aggregate"]
        delta = {}
        if mode != "correct":
            correct_agg = control_report["correct"]["aggregate"]
            delta = {
                "delta_waveform_correlation": correct_agg["waveform_correlation_to_oracle"]["mean"] - agg["waveform_correlation_to_oracle"]["mean"],
                "delta_variance_ratio": correct_agg["variance_ratio"]["mean"] - agg["variance_ratio"]["mean"],
                "delta_cosine": correct_agg["cosine"]["mean"] - agg["cosine"]["mean"],
                "delta_mse": agg["mse"]["mean"] - correct_agg["mse"]["mean"],
            }
        payload = {f"controls/{mode}/{metric}_{stat}": agg[metric][stat] for metric in agg for stat in agg[metric]}
        payload.update({f"controls/{mode}/{k}": v for k, v in delta.items()})
        run.log(payload, step=steps)

    # ---- Gate 4 decision rule ----
    # "Materially outperforms": correct conditioning's mean waveform correlation to the EnCodec
    # oracle must exceed each control's by at least 0.05 absolute (documented threshold, not
    # CLAUDE.md's original 20%-relative `control_margin` -- that margin was defined for a
    # monotonically-positive loss value; waveform correlation is bounded in [-1,1] and can be
    # near zero, where a relative margin is undefined/unstable). Variance ratio must not indicate
    # collapse (kept close to correct conditioning's own value, not literally 1.0 -- per
    # CLAUDE.md's own guidance that this diagnostic identifies collapse, not "correctness").
    CORRELATION_MARGIN = 0.05
    correct_corr = control_report["correct"]["aggregate"]["waveform_correlation_to_oracle"]["mean"]
    correct_var_ratio = control_report["correct"]["aggregate"]["variance_ratio"]["mean"]
    outperformed: dict[str, bool] = {}
    for mode in ("shuffled_across", "shuffled_time", "constant_zero"):
        mode_corr = control_report[mode]["aggregate"]["waveform_correlation_to_oracle"]["mean"]
        outperformed[mode] = (correct_corr - mode_corr) >= CORRELATION_MARGIN
    controls_ok = all(outperformed.values())
    variance_ok = correct_var_ratio > 0.05  # not collapsed to near-zero
    gate4_passed = resume_ok and controls_ok and variance_ok

    print("\n--- Gate 4 decision ---")
    print(f"  correct conditioning wav_corr_to_oracle mean: {correct_corr:.4f}")
    for mode, ok in outperformed.items():
        mode_corr = control_report[mode]["aggregate"]["waveform_correlation_to_oracle"]["mean"]
        print(f"  correct ({correct_corr:.4f}) vs {mode} ({mode_corr:.4f}): "
              f"{'materially better' if ok else 'NOT materially better'} (margin {CORRELATION_MARGIN})")
    print(f"  variance ratio (correct): {correct_var_ratio:.4f} ({'OK' if variance_ok else 'COLLAPSED'})")
    print(f"  checkpoint/resume equivalence: {'PASSED' if resume_ok else 'FAILED'}")
    print(f"\n  GATE 4: {'PASSED' if gate4_passed else 'NOT FULLY PASSED'}")

    run.log({
        "gate4/controls_ok": float(controls_ok),
        "gate4/variance_ok": float(variance_ok),
        "gate4/resume_ok": float(resume_ok),
        "gate4/passed": float(gate4_passed),
        "gate4/correlation_margin": CORRELATION_MARGIN,
    })

    # ---- Listening artifacts: test-split examples, all conditions + oracle + original ----
    print("\n--- Saving listening artifacts (test split) ---")
    listen_rng = random.Random(SEED)
    for label, ex in examples.items():
        if ex.spec.split != "test":
            continue
        with torch.no_grad():
            oracle_audio = encodec.decode(ex.target.transpose(1, 2)).squeeze(0).cpu()
        _write_wav(artifact_root / f"{label}_original.wav", ex.waveform.squeeze(0).cpu(), SAMPLE_RATE)
        _write_wav(artifact_root / f"{label}_encodec_oracle.wav", oracle_audio, SAMPLE_RATE)
        for mode in CONDITIONING_MODES:
            with torch.no_grad():
                conditioned = build_conditioning(mode, ex, all_examples, frame_ratio, listen_rng)
                pred = model(conditioned)
                pred_audio = encodec.decode(pred.transpose(1, 2)).squeeze(0).cpu()
            _write_wav(artifact_root / f"{label}_{mode}.wav", pred_audio, SAMPLE_RATE)
            run.log({f"audio/{label}_{mode}": wandb.Audio(pred_audio.numpy(), sample_rate=SAMPLE_RATE, caption=f"{label} | {mode}")})
        print(f"  wrote listening artifacts for {label}")

    # Best/median/worst under correct conditioning, per this project's existing convention.
    ranking = rank_examples_by_metric(control_report["correct"]["per_example"])
    worst_label, median_label, best_label = ranking[0], ranking[len(ranking) // 2], ranking[-1]
    print(f"  best/median/worst (correct conditioning): {best_label} / {median_label} / {worst_label}")

    train_time_note = "training performed inline above (uninterrupted correct-conditioning run)"

    report = {
        "task": "TASK-13",
        "gate": "Gate 4 revalidation",
        "config": config,
        "resume_check": {
            "predeterminism_max_relative_diff": max_predeterminism_diff,
            "resume_max_relative_diff": max_resume_diff,
            "tolerance": RESUME_TOLERANCE,
            "determinism_ok": determinism_ok,
            "resume_ok": resume_ok,
        },
        "parameter_count": param_count,
        "receptive_field_frames": model.receptive_field_frames,
        "final_val_mse": final_val_mse,
        "best_val_mse_during_training": best_val_state,
        "control_report": control_report,
        "gate4_decision": {
            "correlation_margin": CORRELATION_MARGIN,
            "correct_waveform_correlation_mean": correct_corr,
            "correct_variance_ratio_mean": correct_var_ratio,
            "controls_outperformed": outperformed,
            "controls_ok": controls_ok,
            "variance_ok": variance_ok,
            "resume_ok": resume_ok,
            "passed": gate4_passed,
        },
        "best_median_worst_correct": {"best": best_label, "median": median_label, "worst": worst_label},
        "note": train_time_note,
    }
    report_path = artifact_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {report_path}")

    log_artifact_file(run, report_path, "report-task13-gate4-encodec", "report")
    log_artifact_file(run, final_ckpt_path, "checkpoint-task13-gate4-encodec-final", "model")
    run.log({"wandb_run_id": run.id})
    run.finish()
    print(f"W&B run finished: {run.id}")
    print(f"\nFINAL: Gate 4 {'PASSED' if gate4_passed else 'NOT FULLY PASSED'} -- see {report_path}")

    if not gate4_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
