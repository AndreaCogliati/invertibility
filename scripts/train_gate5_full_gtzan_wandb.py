"""TASK-14: Gate 5 full-GTZAN pilot.

    frozen MuQ last_hidden_state -> repeat_interleave to EnCodec's frame rate
        -> SpectrogramToDacTranslator (unmodified) -> frozen EnCodec latent
        -> frozen EnCodec decoder -> reconstructed audio

Trains on the full committed GTZAN training split (792 tracks), sampling broadly across all of
them via a deterministic, resumable random-crop sampler (`gate5_common.TrainCropSampler`) rather
than rotating over a small fixed set of clips (TASK-13's 16-crop limitation). Held-out validation
and test crops are a fixed, versioned manifest (`data/manifests/gate5_heldout_crops.csv`, built by
`scripts/build_gate5_manifests.py`), never touched during training. Four conditioning controls
(`correct`/`shuffled_across`/`shuffled_time`/`constant_zero`) are evaluated against the single
best-by-validation checkpoint at the end, exactly as in TASK-13 -- reusing
`gate4_encodec_common.py`'s manifest loader and control constructions unchanged.

Per this task's explicit "do not run the full training job locally" / "do not submit the full
BlueHive job unless explicitly instructed" constraints: this script is executed here ONLY with
`configs/gate5_smoke.yaml` (a tiny, clearly-labeled bounded diagnostic, not scientific evidence).
The full run (`configs/gate5_full_gtzan.yaml`) is submitted via `bluehive/run_task14.slurm` only
when explicitly instructed.

Findings and full write-up: notes/DECISIONS.md#task14-gate5-full-gtzan-pilot, notes/SCRATCHPAD.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

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
    build_conditioning,
    load_gate4_examples,
)
from muq_inversion.gate5_common import (
    Gate5Config,
    TrainCropSampler,
    compute_held_out_audio_metrics,
    load_gate5_config,
    load_train_crop_waveform,
    load_train_tracks,
    paired_delta_stats,
    write_limited_heldout_manifest,
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
MUQ_CHANNELS = 1024
ENCODEC_CHANNELS = 128

STFT_FFT_SIZES = (512, 1024, 2048)
STFT_HOP_SIZES = (50, 120, 240)
STFT_WIN_LENGTHS = (240, 600, 1200)


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
    """Mirrors `scripts/train_gate4_encodec_wandb.py`'s `_enable_cuda_determinism` fix
    (TASK-13, commit `895b81b`), confirmed on BlueHive (NVIDIA A40). Duplicated rather than
    imported: per this task's "preserve TASK-13 code" instruction, TASK-13's script is not
    modified/refactored to share this helper."""
    if device.type != "cuda":
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model_and_optimizer(
    config: Gate5Config, device: torch.device
) -> tuple[SpectrogramToDacTranslator, torch.optim.Optimizer]:
    torch.manual_seed(config.seed)
    model = SpectrogramToDacTranslator(input_channels=MUQ_CHANNELS, dac_channels=ENCODEC_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=tuple(config.adam_betas))
    return model, optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a gate5_*.yaml config")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument(
        "--resume", choices=["auto", "none"], default="none",
        help="'auto' resumes from runs/<run_dir_name>/checkpoints/latest.pt if present",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_gate5_config(Path(args.config))
    print(f"loaded config: {args.config} (run_label={config.run_label})")

    try:
        device = select_device(requested=config.device_request, allow_cpu_fallback=config.allow_cpu_fallback)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")
    _enable_cuda_determinism(device)
    environment = capture_environment_manifest(REPO_ROOT)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    muq = FrozenMuQEncoder(config.muq_model_id, device)
    encodec = FrozenEncodecEncoder(config.encodec_model_id, device, bandwidth=config.encodec_bandwidth_kbps)

    # ---- Dataset and split verification (this task's explicit "at runtime" requirements) ----
    full_manifest_path = REPO_ROOT / config.full_gtzan_manifest
    manifest_hash = _file_hash(full_manifest_path)
    print(f"full GTZAN manifest hash: {manifest_hash}")

    train_tracks = load_train_tracks(full_manifest_path)
    print(f"valid train tracks: {len(train_tracks)}")
    if config.train_track_limit is not None:
        train_tracks = train_tracks[: config.train_track_limit]
        print(f"  bounded to {len(train_tracks)} tracks (smoke-test config)")

    # ---- Held-out manifest: fixed, versioned, never touched during training ----
    heldout_manifest_path = REPO_ROOT / config.heldout_manifest
    if config.heldout_crop_limit is not None:
        heldout_manifest_path = (
            REPO_ROOT / "runs" / config.run_dir_name / "limited_heldout_crops.csv"
        )
        write_limited_heldout_manifest(
            REPO_ROOT / config.heldout_manifest, heldout_manifest_path, config.heldout_crop_limit
        )
        print(f"  bounded held-out manifest written to {heldout_manifest_path} (smoke-test config)")
    heldout_manifest_hash = _file_hash(heldout_manifest_path)

    print("precomputing MuQ/EnCodec latents for the fixed held-out crop set ...")
    heldout_examples, frame_ratio = load_gate4_examples(
        heldout_manifest_path, "gate5", muq, encodec, config.sample_rate, device
    )
    val_labels = sorted(lbl for lbl, ex in heldout_examples.items() if ex.spec.split == "val")
    test_labels = sorted(lbl for lbl, ex in heldout_examples.items() if ex.spec.split == "test")
    print(f"held-out: {len(val_labels)} val crops, {len(test_labels)} test crops, frame_ratio={frame_ratio}")
    if not val_labels or not test_labels:
        raise SystemExit("STOP CONDITION: held-out manifest must contain both val and test crops")

    raw_dir = REPO_ROOT / "data/raw/gtzan"

    manifest_hash_full = manifest_hash  # for the resolved-config record below

    resolved_config = config.to_dict()
    resolved_config.update(
        {
            "git_commit": environment.get("git_commit"),
            "git_dirty": environment.get("git_dirty"),
            "full_gtzan_manifest_hash": manifest_hash_full,
            "heldout_manifest_hash": heldout_manifest_hash,
            "uv_lock_hash": _file_hash(REPO_ROOT / "uv.lock"),
            "frame_ratio": frame_ratio,
            "num_train_tracks": len(train_tracks),
            "num_val_crops": len(val_labels),
            "num_test_crops": len(test_labels),
            "device": str(device),
            "wandb_version": wandb.__version__,
        }
    )

    run_root = REPO_ROOT / "runs" / config.run_dir_name
    artifact_root = REPO_ROOT / "artifacts" / config.artifact_dir_name
    checkpoint_dir = run_root / "checkpoints"
    latest_ckpt_path = checkpoint_dir / "latest.pt"

    # Peek at any existing checkpoint's W&B run ID before init_run, so a resumed job reopens the
    # SAME W&B run (tasks/TASK-14.md's "resumption must append to the same scientific run
    # lineage" requirement) rather than starting a new, disconnected one.
    resume_wandb_id: str | None = None
    if args.resume == "auto" and latest_ckpt_path.exists():
        peek = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        resume_wandb_id = peek.get("wandb_run_id")
        if resume_wandb_id is None:
            print("  WARNING: existing checkpoint has no wandb_run_id (pre-dates this field) -- starting a new W&B run")

    run = init_run(
        config=resolved_config,
        environment=environment,
        run_name=make_run_name(prefix=config.run_label),
        mode=args.wandb_mode,
        group=config.wandb_group,
        job_type="gate5-full-gtzan-pilot",
        tags=list(config.wandb_tags),
        resume_id=resume_wandb_id,
    )
    print(f"W&B run: {run.id} ({'resumed' if resume_wandb_id else 'new'})")
    run.log({"slurm_job_id": environment.get("slurm_job_id")})

    model, optimizer = build_model_and_optimizer(config, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"parameters: {param_count:,}  receptive_field: {model.receptive_field_frames} frames")

    sampler = TrainCropSampler(
        tracks=train_tracks, crop_duration_seconds=config.crop_duration_seconds, rng=random.Random(config.seed)
    )
    mse_loss_fn = nn.MSELoss()
    stft_loss_fn = MultiResolutionSTFTLoss(STFT_FFT_SIZES, STFT_HOP_SIZES, STFT_WIN_LENGTHS).to(device)

    def evaluate_example(m: SpectrogramToDacTranslator, ex, conditioned: torch.Tensor) -> dict[str, float]:
        """Combines `evaluate_prediction` (latent MSE/cosine/norm/variance ratio, lag-aligned
        translator-vs-oracle waveform correlation and STFT distance) with
        `compute_held_out_audio_metrics` (the three original-anchored waveform correlations and
        `oracle_relative_waveform_recovery` -- `tasks/TASK-14.md`'s explicit checkpoint-selection
        and primary-endpoint metric, which `evaluate_prediction` alone does not provide: its own
        `waveform_correlation_to_oracle` compares the translator's audio to the oracle's audio
        directly, not each of them independently to the original)."""
        with torch.no_grad():
            pred = m(conditioned)
        metrics = evaluate_prediction(
            pred, ex.target, dac=encodec, stft_loss_fn=stft_loss_fn, device=device, compute_audio_metrics=True
        )
        with torch.no_grad():
            oracle_audio = encodec.decode(ex.target.transpose(1, 2)).squeeze(0)
            pred_audio = encodec.decode(pred.transpose(1, 2)).squeeze(0)
        metrics.update(
            compute_held_out_audio_metrics(ex.waveform.squeeze(0), oracle_audio, pred_audio, stft_loss_fn, device)
        )
        return metrics

    # ---- Resume ----
    start_step = 0
    best_metric = -float("inf")
    best_step = 0
    if args.resume == "auto" and latest_ckpt_path.exists():
        print(f"resuming from {latest_ckpt_path}")
        loaded = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        if loaded["manifest_hash"] != manifest_hash_full:
            raise SystemExit(
                f"STOP CONDITION: checkpoint's full-GTZAN manifest hash {loaded['manifest_hash']} "
                f"!= current {manifest_hash_full} -- refusing to resume against a different split"
            )
        model.load_state_dict(loaded["model_state_dict"])
        optimizer.load_state_dict(loaded["optimizer_state_dict"])
        sampler.load_state_dict(loaded["sampler_state"])
        _rng_restore(loaded["rng_state"])
        start_step = loaded["step"]
        best_metric = loaded.get("best_metric", best_metric)
        best_step = loaded.get("best_step", best_step)
        print(f"  resumed at step {start_step}, best_metric so far {best_metric:.4f} (step {best_step})")

    def eval_val_correct_only() -> dict[str, dict[str, float]]:
        """Periodic validation: correct conditioning only, val split only."""
        per_example = {}
        model.eval()
        for label in val_labels:
            ex = heldout_examples[label]
            conditioned = ex.muq.repeat_interleave(frame_ratio, dim=1)
            per_example[label] = evaluate_example(model, ex, conditioned)
        model.train()
        return per_example

    train_start = time.perf_counter()
    step = start_step
    while step < config.steps:
        step += 1
        draw = sampler.draw()
        crop = load_train_crop_waveform(draw, raw_dir, config.sample_rate).to(device)

        muq_feat = muq.encode(crop).clone()
        target = encodec.encode(crop).transpose(1, 2).clone()
        if not torch.isfinite(muq_feat).all() or not torch.isfinite(target).all():
            raise SystemExit(f"STOP CONDITION: non-finite MuQ/EnCodec latent at step {step} ({draw.track_id})")

        repeated = muq_feat.repeat_interleave(frame_ratio, dim=1)
        predicted = model(repeated)
        mse = mse_loss_fn(predicted, target)
        cos_sim = nn.functional.cosine_similarity(predicted, target, dim=-1).mean()
        total_loss = mse + config.cosine_loss_weight * (1.0 - cos_sim)

        if not torch.isfinite(total_loss):
            raise SystemExit(f"STOP CONDITION: non-finite loss at step {step}: {total_loss.item()}")

        optimizer.zero_grad()
        total_loss.backward()
        grad_norms = three_way_grad_norms(model)
        all_grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        global_grad_norm = float(sum(g.norm(2) ** 2 for g in all_grads) ** 0.5)
        nonfinite_grads = count_nonfinite(all_grads)
        optimizer.step()

        if step % config.log_every == 0 or step == config.steps:
            elapsed = time.perf_counter() - train_start
            exposure = sampler.exposure_summary()
            run.log(
                {
                    "train/total_loss": total_loss.item(),
                    "train/mse": mse.item(),
                    "train/cosine": cos_sim.item(),
                    "train/learning_rate": config.learning_rate,
                    "grad/global_l2_norm": global_grad_norm,
                    "grad/mel_projection_l2_norm": grad_norms["mel_projection_l2_norm"],
                    "grad/translator_l2_norm": grad_norms["translator_l2_norm"],
                    "grad/output_layer_l2_norm": grad_norms["output_layer_l2_norm"],
                    "grad/nonfinite_count": nonfinite_grads,
                    "throughput/steps_per_second": step / elapsed if elapsed > 0 else 0.0,
                    "throughput/elapsed_seconds": elapsed,
                    "sampler/tracks_visited": exposure["tracks_visited"],
                    "sampler/tracks_total": exposure["tracks_total"],
                    "sampler/min_exposure": exposure["min_exposure"],
                    "sampler/max_exposure": exposure["max_exposure"],
                    "sampler/mean_exposure": exposure["mean_exposure"],
                    "sampler/num_draws": sampler.num_draws,
                },
                step=step,
            )
            print(f"  step {step}/{config.steps}: loss={total_loss.item():.4f} mse={mse.item():.4f} "
                  f"cos={cos_sim.item():.4f} tracks_visited={exposure['tracks_visited']}/{exposure['tracks_total']}")

        is_val_step = step % config.val_every == 0 or step == config.steps
        if is_val_step:
            per_example = eval_val_correct_only()
            agg = aggregate_metrics_across_examples(per_example)
            val_metric = agg["oracle_relative_waveform_recovery"]["mean"]  # tasks/TASK-14.md's explicit checkpoint-selection metric
            log_payload = {f"val/distribution/{m}_{s}": agg[m][s] for m in agg for s in agg[m]}
            log_payload["val/checkpoint_selection_metric"] = val_metric
            run.log(log_payload, step=step)
            print(f"    val (correct-conditioning) wav_corr_to_oracle mean={val_metric:.4f}")

            if val_metric > best_metric:
                best_metric = val_metric
                best_step = step
                _save_atomic(
                    checkpoint_dir / "best.pt",
                    {
                        "step": step, "model_state_dict": model.state_dict(),
                        "best_metric": best_metric, "best_step": best_step,
                        "manifest_hash": manifest_hash_full, "resolved_config": resolved_config,
                        "wandb_run_id": run.id,
                    },
                )
            run.log({"val/best_metric_so_far": best_metric, "val/best_step_so_far": best_step}, step=step)

        is_milestone = (
            step == config.early_diagnostic_step
            or step in config.intermediate_checkpoint_steps
            or step == config.steps
        )
        if is_milestone or step % config.checkpoint_every == 0:
            state = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "sampler_state": sampler.state_dict(),
                "rng_state": _rng_snapshot(),
                "best_metric": best_metric,
                "best_step": best_step,
                "manifest_hash": manifest_hash_full,
                "heldout_manifest_hash": heldout_manifest_hash,
                "uv_lock_hash": resolved_config["uv_lock_hash"],
                "git_commit": environment.get("git_commit"),
                "git_dirty": environment.get("git_dirty"),
                "resolved_config": resolved_config,
                "wandb_run_id": run.id,
            }
            _save_atomic(latest_ckpt_path, state)
            if is_milestone:
                _save_atomic(checkpoint_dir / f"step{step}.pt", state)

    train_elapsed = time.perf_counter() - train_start
    print(f"\nfinished: {config.steps} steps, {train_elapsed:.2f}s")

    # ---- Final evaluation: best checkpoint, all four conditions, val + test separately ----
    print("\n--- Final held-out conditioning-control evaluation (best checkpoint) ---")
    best_state = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    eval_model = SpectrogramToDacTranslator(input_channels=MUQ_CHANNELS, dac_channels=ENCODEC_CHANNELS).to(device)
    eval_model.load_state_dict(best_state["model_state_dict"])
    eval_model.eval()

    all_heldout = list(heldout_examples.values())
    control_rng = random.Random(config.seed)
    control_report: dict[str, dict] = {}
    for mode in CONDITIONING_MODES:
        per_example = {}
        for label, ex in heldout_examples.items():
            conditioned = build_conditioning(mode, ex, all_heldout, frame_ratio, control_rng)
            per_example[label] = evaluate_example(eval_model, ex, conditioned)
        control_report[mode] = {
            "per_example": per_example,
            "val_aggregate": aggregate_metrics_across_examples({lbl: per_example[lbl] for lbl in val_labels}),
            "test_aggregate": aggregate_metrics_across_examples({lbl: per_example[lbl] for lbl in test_labels}),
        }
        run.log(
            {
                f"controls/{mode}/val/{m}_{s}": v
                for m, stats in control_report[mode]["val_aggregate"].items()
                for s, v in stats.items()
            }
        )
        run.log(
            {
                f"controls/{mode}/test/{m}_{s}": v
                for m, stats in control_report[mode]["test_aggregate"].items()
                for s, v in stats.items()
            }
        )
        print(f"  {mode}: val wav_corr={control_report[mode]['val_aggregate']['waveform_correlation_to_oracle']['mean']:.4f} "
              f"test wav_corr={control_report[mode]['test_aggregate']['waveform_correlation_to_oracle']['mean']:.4f}")

    # ---- Paired deltas (held-out only, never pooled with train -- there is no train split in
    # heldout_examples). Primary endpoint per this task's explicit instruction:
    # "the paired held-out difference in oracle-relative waveform recovery". Waveform correlation
    # to the oracle (translator-vs-oracle audio, not anchored to the original) is retained as a
    # secondary paired-delta metric.
    PRIMARY_METRIC = "oracle_relative_waveform_recovery"
    SECONDARY_METRIC = "waveform_correlation_to_oracle"
    paired_deltas: dict[str, dict[str, dict]] = {}
    for split_name, labels in (("val", val_labels), ("test", test_labels)):
        paired_deltas[split_name] = {}
        correct_per_example = control_report["correct"]["per_example"]
        for mode in ("shuffled_across", "shuffled_time", "constant_zero"):
            control_per_example = control_report[mode]["per_example"]
            paired_deltas[split_name][mode] = {
                metric: paired_delta_stats(
                    [correct_per_example[lbl][metric] for lbl in labels],
                    [control_per_example[lbl][metric] for lbl in labels],
                    seed=config.seed,
                )
                for metric in (PRIMARY_METRIC, SECONDARY_METRIC)
            }

    # ---- Outcome classification (A-F), per this task's explicit rubric ----
    def _materially_better(split_name: str) -> dict[str, bool]:
        return {
            mode: paired_deltas[split_name][mode][PRIMARY_METRIC]["mean_delta"] > 0.05
            and paired_deltas[split_name][mode][PRIMARY_METRIC]["proportion_positive"] > 0.5
            for mode in ("shuffled_across", "shuffled_time", "constant_zero")
        }

    val_better = _materially_better("val")
    test_better = _materially_better("test")
    val_all = all(val_better.values())
    test_all = all(test_better.values())
    val_var = control_report["correct"]["val_aggregate"]["variance_ratio"]["mean"]
    collapsed = val_var <= 0.05

    if collapsed:
        outcome = "E"  # collapse/instability
    elif val_all and test_all:
        outcome = "A"  # held-out conditioning generalization
    elif any(val_better.values()) or any(test_better.values()):
        outcome = "B"  # partial generalization
    else:
        outcome = "D"  # no held-out conditioning dependence

    gate5_passed = val_all and test_all and not collapsed

    print("\n--- Gate 5 decision ---")
    print(f"  outcome classification: {outcome}")
    print(f"  val: correct materially beats all controls: {val_all} ({val_better})")
    print(f"  test: correct materially beats all controls: {test_all} ({test_better})")
    print(f"  variance ratio (correct, val): {val_var:.4f} ({'collapsed' if collapsed else 'OK'})")
    print(f"  GATE 5: {'PASSED' if gate5_passed else 'NOT PASSED'}")

    run.log(
        {
            "gate5/outcome": outcome,
            "gate5/val_all_controls_beaten": float(val_all),
            "gate5/test_all_controls_beaten": float(test_all),
            "gate5/variance_collapsed": float(collapsed),
            "gate5/passed": float(gate5_passed),
        }
    )

    # ---- Listening artifacts: fixed representative subset ----
    print("\n--- Saving listening artifacts ---")
    listen_rng = random.Random(config.seed)
    val_subset = val_labels[: config.listening_subset_size]
    test_subset = test_labels[: config.listening_subset_size]
    for label in val_subset + test_subset:
        ex = heldout_examples[label]
        with torch.no_grad():
            oracle_audio = encodec.decode(ex.target.transpose(1, 2)).squeeze(0).cpu()
        _write_wav(artifact_root / f"{label}_original.wav", ex.waveform.squeeze(0).cpu(), config.sample_rate)
        _write_wav(artifact_root / f"{label}_encodec_oracle.wav", oracle_audio, config.sample_rate)
        for mode in CONDITIONING_MODES:
            with torch.no_grad():
                conditioned = build_conditioning(mode, ex, all_heldout, frame_ratio, listen_rng)
                pred_audio = encodec.decode(eval_model(conditioned).transpose(1, 2)).squeeze(0).cpu()
            _write_wav(artifact_root / f"{label}_{mode}.wav", pred_audio, config.sample_rate)
        print(f"  wrote listening artifacts for {label}")

    ranking = rank_examples_by_metric(control_report["correct"]["per_example"])
    worst_label, median_label, best_label = ranking[0], ranking[len(ranking) // 2], ranking[-1]

    report = {
        "task": "TASK-14",
        "gate": config.gate,
        "run_label": config.run_label,
        "resolved_config": resolved_config,
        "parameter_count": param_count,
        "receptive_field_frames": eval_model.receptive_field_frames,
        "best_checkpoint_step": best_step,
        "best_checkpoint_val_metric": best_metric,
        "final_step": config.steps,
        "training_time_seconds": train_elapsed,
        "sampler_exposure_summary": sampler.exposure_summary(),
        "control_report": control_report,
        "paired_deltas": paired_deltas,
        "outcome_classification": outcome,
        "gate5_decision": {
            "val_all_controls_beaten": val_all,
            "test_all_controls_beaten": test_all,
            "variance_collapsed": collapsed,
            "passed": gate5_passed,
        },
        "best_median_worst_correct": {"best": best_label, "median": median_label, "worst": worst_label},
    }
    report_path = artifact_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {report_path}")

    log_artifact_file(run, report_path, f"report-{config.run_label}", "report")
    log_artifact_file(run, checkpoint_dir / "best.pt", f"checkpoint-{config.run_label}-best", "model")
    run.log({"wandb_run_id": run.id})
    run.finish()
    print(f"W&B run finished: {run.id}")
    print(f"\nFINAL: Gate 5 {'PASSED' if gate5_passed else 'NOT PASSED'} (outcome {outcome}) -- see {report_path}")

    if not gate5_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
