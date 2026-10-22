"""TASK-18: 32/8/8 held-out generalization of the corrected MuQ -> BigVGAN-mel -> BigVGAN pipeline.

    audio -> frozen MuQ last_hidden_state -> frame-center alignment -> MuQToMelTranslator
          -> raw 100-band log-mel -> clamp to BigVGAN's log floor (decode only) -> frozen BigVGAN-v2

    uv run python scripts/train_task18_heldout.py --device xpu --run-name task18_heldout_v1 --wandb-mode offline
    uv run python scripts/train_task18_heldout.py --config configs/task18_smoke.yaml --device mps --wandb-mode disabled

Phases, all resumable/idempotent:
  1. MuQ features + official target mels for every manifest crop (content-addressed cache).
  2. Training on the 768-crop pool with track-balanced minibatches; every `eval_every` steps, mel
     metrics on memorized / train_unseen / val crops. Best checkpoint = lowest val mel L1. Test
     crops are not touched during training.
  3. Full evaluation of the best checkpoint (all four regimes x all conditions, mel + decoded-audio
     metrics, listening WAVs) and mel-level evaluation of the final checkpoint.
  4. report.json, per_example.csv, TASK-18_RESULTS.md (with the pre-registered outcome).

`--resume auto` (default) continues an existing run directory from `checkpoints/latest.pt`
(model, Adam, sampler, RNG, best-so-far, W&B run id); a finished run is never overwritten.
SIGTERM/SIGINT save `latest.pt` at the end of the current step and exit with code 3.

TASK-19 reuses this script; TASK-18 configs behave exactly as before. TASK-19 configs add `arms`
(selected with `--arm`): which training tracks form the pool (`train_tracks: core|all`), the
learning-rate schedule (`lr_schedule`), extra evaluation sets (`eval_sets`), and an
evaluation-only mode for existing checkpoints (`evaluate_checkpoints`). Every default reproduces
TASK-18.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import signal
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml

from muq_inversion.audio import load_resample_validate
from muq_inversion.bigvgan_common import (
    BIGVGAN_DIR,
    PEAK_NORMALIZE_TARGET,
    audio_metrics,
    bigvgan_log_mel_floor,
    load_bigvgan,
    official_mel,
)
from muq_inversion.device import (
    DEVICE_CHOICES,
    backend_stage,
    device_metadata,
    load_portable,
    memory_stats,
    require_fp32_default,
    seed_everything,
)
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.logging_utils import capture_environment_manifest, select_device
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.models.muq_mel_translator import MuQToMelTranslator
from muq_inversion.task18 import (
    CONDITIONS,
    EVAL_SETS,
    TrackBalancedSampler,
    classify_outcome,
    cluster_bootstrap_delta,
    conditioned_features,
    contour_metrics,
    file_sha256,
    mel_metrics,
    shuffled_across_index,
    shuffled_time_perms,
)
from muq_inversion.task18_report import render_results
from muq_inversion.task19 import lr_multiplier
from muq_inversion.task20 import early_stopping_update
from muq_inversion.time_alignment import (
    bigvgan_mel_frame_centers,
    muq_token_centers,
    source_positions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/task18_heldout.yaml"
STFT = ((512, 1024, 2048), (50, 120, 240), (240, 600, 1200))
FEATURE_CACHE_VERSION = 1
HIGHER_IS_BETTER = ("mel_cosine", "mel_pearson", "rms_corr", "centroid_corr", "chroma_cosine")


def load_config(path: Path) -> dict:
    """YAML config; `base_config` (relative to the repo root) is loaded first, recursively, and
    overridden key by key (top level only)."""
    cfg = yaml.safe_load(Path(path).read_text())
    if "base_config" in cfg:
        cfg = {**load_config(REPO_ROOT / cfg.pop("base_config")), **cfg}
    return cfg


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def apply_smoke_limits(rows: list[dict], limits: dict | None, memorized: list[int]) -> list[dict]:
    if not limits:
        return rows
    keep: set[str] = set()
    groups: dict[tuple, set] = {}
    for r in rows:  # TASK-18 rows: one group per split; TASK-19/20 rows: also by core/tier/cohort
        groups.setdefault((r["split"], r.get("core", ""), r.get("tier", ""), r.get("cohort", "")), set()).add(r["track_id"])
    for (split, *_), ids in groups.items():
        n = limits["train_tracks"] if split == "train" else limits["heldout_tracks_per_split"]
        if split == "train" and len([g for g in groups if g[0] == "train"]) > 1:
            n = max(1, n // 2)
        keep |= set(sorted(ids)[:n])
    out = []
    for r in rows:
        if r["track_id"] not in keep:
            continue
        if r["role"] == "train_pool":
            idx = int(r["crop_id"].rsplit(":", 1)[1])
            if idx >= limits["pool_crops_per_track"] and idx not in memorized:
                continue
        out.append(r)
    return out


def atomic_torch_save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def slurm_resources() -> dict:
    keys = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT", "SLURM_NTASKS",
            "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE",
            "SLURM_JOB_NODELIST", "SLURM_SUBMIT_DIR", "SLURM_RESTART_COUNT")
    return {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}


# --- phase 1: features -------------------------------------------------------------------

def build_features(rows: list[dict], cfg: dict, h, muq: FrozenMuQEncoder | None, device: torch.device,
                   cache_key_parts: dict) -> tuple[list[tuple[str, int]], dict[str, torch.Tensor], Path]:
    """Unique (path, start) crops -> MuQ last_hidden_state (CPU float32), official target mel,
    and the peak-normalized waveform, cached atomically under a content-addressed name."""
    keys = sorted({(r["path"], int(r["crop_start_sample"])) for r in rows})
    digest = hashlib.sha256(json.dumps({**cache_key_parts, "keys": keys}, sort_keys=True).encode()).hexdigest()[:20]
    cache = REPO_ROOT / cfg["feature_cache_dir"] / f"task18_v{FEATURE_CACHE_VERSION}_{digest}.pt"
    if cache.exists():
        data = torch.load(cache, map_location="cpu", weights_only=False)
        if data["keys"] == keys:
            print(f"feature cache hit: {cache.relative_to(REPO_ROOT)}")
            return keys, data["tensors"], cache
    if muq is None:
        raise SystemExit("feature cache missing and MuQ not loaded")
    n = cfg["num_mel_hops"] * cfg["hop_size"]
    waves = []
    by_path: dict[str, list[int]] = {}
    for p, s in keys:
        by_path.setdefault(p, []).append(s)
    wave_of = {}
    for p, starts in by_path.items():
        track, _ = load_resample_validate(REPO_ROOT / p, h.sampling_rate)
        track = librosa.util.normalize(track) * PEAK_NORMALIZE_TARGET  # identical to load_mel_crops
        for s in starts:
            if s + n > len(track):
                raise SystemExit(f"STOP: {p} crop at {s} exceeds {len(track)} samples")
            wave_of[(p, s)] = torch.from_numpy(np.ascontiguousarray(track[s:s + n])).float()
    waves = torch.stack([wave_of[k] for k in keys])
    mels = torch.cat([official_mel(waves[i:i + 64], h) for i in range(0, len(keys), 64)])
    if mels.shape[-1] != cfg["num_mel_hops"]:
        raise SystemExit(f"STOP: target mel has {mels.shape[-1]} frames")
    feats = []
    t0 = time.perf_counter()
    with backend_stage("MuQ feature extraction", device):
        for i in range(0, len(keys), 16):
            feats.append(muq.encode(waves[i:i + 16].to(device)).float().cpu().clone())
    feats = torch.cat(feats)
    print(f"MuQ features for {len(keys)} crops in {time.perf_counter() - t0:.1f}s: {tuple(feats.shape)}")
    if not (torch.isfinite(feats).all() and torch.isfinite(mels).all()):
        raise SystemExit("STOP CONDITION: non-finite MuQ features or target mel")
    tensors = {"muq": feats, "mel": mels, "wave": waves}
    atomic_torch_save(cache, {"keys": keys, "tensors": tensors, "key_parts": cache_key_parts})
    check = torch.load(cache, map_location="cpu", weights_only=False)
    assert check["keys"] == keys and torch.equal(check["tensors"]["muq"], feats), "cache round-trip failed"
    return keys, tensors, cache


# --- main --------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default=None)
    parser.add_argument("--run-name", default=None, help="run directory name (default: prefix + UTC time)")
    parser.add_argument("--resume", choices=["auto", "never"], default="auto")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--arm", default=None, help="TASK-19: arm name from the config's `arms`")
    parser.add_argument("--evaluate-checkpoints", default=None,
                        help="override the arm's evaluate_checkpoints dir (evaluation-only; e.g. smoke tests)")
    parser.add_argument("--stop-after-step", type=int, default=None,
                        help="simulate an interruption: save latest.pt after this step and exit 3 (resume test)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.arm is not None:
        if args.arm not in cfg.get("arms", {}):
            raise SystemExit(f"--arm {args.arm} not in {args.config} arms {sorted(cfg.get('arms', {}))}")
        cfg = {**cfg, **cfg["arms"][args.arm], "arm": args.arm}
        cfg["run_name_prefix"] = f"{cfg['run_name_prefix']}_{args.arm}"
    eval_sets = list(cfg.get("eval_sets", EVAL_SETS))
    if eval_sets[:4] != list(EVAL_SETS):
        raise SystemExit(f"eval_sets must start with {EVAL_SETS} (TASK-18 regimes, same control seeds)")
    if args.evaluate_checkpoints:
        cfg["evaluate_checkpoints"] = args.evaluate_checkpoints
    evaluate_only = cfg.get("evaluate_checkpoints")
    if cfg.get("temporal_alignment") != "frame_centers" or not cfg.get("decode_clamp_to_mel_floor"):
        raise SystemExit("TASK-18 uses the corrected pipeline: frame_centers alignment + decode clamp")

    run_name = args.run_name or f"{cfg['run_name_prefix']}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = REPO_ROOT / cfg["run_root"] / run_name
    out_dir = REPO_ROOT / cfg["artifact_root"] / run_name
    ckpt_dir = run_dir / "checkpoints"
    latest = ckpt_dir / "latest.pt"
    if (out_dir / "report.json").exists():
        raise SystemExit(f"{out_dir / 'report.json'} exists: run {run_name} is finished and is never overwritten")
    resuming = latest.exists() and args.resume == "auto"
    if run_dir.exists() and not resuming and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} exists without a resumable checkpoint (or --resume never); choose a new --run-name")

    device_request = args.device or cfg["device_request"]
    device = select_device(device_request, cfg["allow_cpu_fallback"])
    require_fp32_default()
    seeding = seed_everything(cfg["seed"], device)
    environment = capture_environment_manifest(REPO_ROOT)
    device_info = device_metadata(device_request, device)
    print(f"run {run_name} | device: {json.dumps(device_info)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    track_path, crop_path = REPO_ROOT / cfg["track_manifest"], REPO_ROOT / cfg["crop_manifest"]
    rows = apply_smoke_limits(read_csv(crop_path), cfg.get("smoke_limits"), cfg["memorized_pool_indices"])
    # Only crops this run uses (training pool + configured eval sets) get features. A no-op for the
    # TASK-18/19 manifests, which contain nothing else.
    rows = [r for r in rows if r["role"] == "train_pool" or r["eval_set"] in eval_sets]
    hashes = {"track_manifest_sha256": file_sha256(track_path), "crop_manifest_sha256": file_sha256(crop_path),
              "uv_lock_sha256": file_sha256(REPO_ROOT / "uv.lock"),
              "bigvgan_generator_sha256": file_sha256(BIGVGAN_DIR / "bigvgan_generator.pt")}
    for t in read_csv(track_path):  # audio integrity: every track still matches its recorded checksum
        if t["track_id"] in {r["track_id"] for r in rows} and file_sha256(REPO_ROOT / t["path"]) != t["checksum_sha256"]:
            raise SystemExit(f"STOP: checksum mismatch for {t['path']}")

    # Run manifest + lineage (appended on every start, including resumes).
    manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "task": cfg["task"], "arm": cfg.get("arm"), "run_name": run_name, "config_path": str(args.config), "resolved_config": cfg,
        "hashes": hashes, "output_paths": {"run_dir": str(run_dir.relative_to(REPO_ROOT)),
                                           "artifact_dir": str(out_dir.relative_to(REPO_ROOT))}, "starts": []}
    if run_manifest["hashes"] != hashes:
        raise SystemExit(f"STOP: manifests/lock/weights changed since this run started: {run_manifest['hashes']} vs {hashes}")
    run_manifest["starts"].append({"utc": datetime.now(UTC).isoformat(), "resuming": resuming, "device": device_info,
                                   "seeding": seeding, "environment": environment, "slurm": slurm_resources(),
                                   "argv": sys.argv})
    atomic_write_text(manifest_path, json.dumps(run_manifest, indent=2, default=str))

    # Frozen models and features.
    with backend_stage("load BigVGAN", device):
        bigvgan = load_bigvgan(device)
    h = bigvgan.h
    with backend_stage("load MuQ", device):
        muq = FrozenMuQEncoder(cfg["muq_model_id"], device)
    cache_parts = {"version": FEATURE_CACHE_VERSION, "muq": cfg["muq_model_id"], "bigvgan": cfg["bigvgan_revision"],
                   "num_hops": cfg["num_mel_hops"], "backend": device.type, "torch": torch.__version__}
    keys, tensors, cache_path = build_features(rows, cfg, h, muq, device, cache_parts)
    t_muq, t_mel = tensors["muq"].shape[1], cfg["num_mel_hops"]
    positions = source_positions(bigvgan_mel_frame_centers(h, t_mel), muq_token_centers(muq.model.model, t_muq)).to(device)
    del muq
    index_of = {k: i for i, k in enumerate(keys)}
    feats_dev = tensors["muq"].to(device)
    mel_dev = tensors["mel"].to(device)
    mel_floor = bigvgan_log_mel_floor()

    def idx(r):
        return index_of[(r["path"], int(r["crop_start_sample"]))]

    pool_by_track: dict[str, list[int]] = {}
    train_tracks = cfg.get("train_tracks", "all")
    for r in rows:
        if r["role"] == "train_pool" and (train_tracks == "all" or r.get("core") == "1"):
            pool_by_track.setdefault(r["track_id"], []).append(idx(r))
    pool_all = sorted({i for v in pool_by_track.values() for i in v})
    avg_mel = tensors["mel"][pool_all].mean(dim=(0, 2))  # (num_mels,) per-bin training-pool mean
    eval_rows = {s: [r for r in rows if r["eval_set"] == s] for s in eval_sets}
    eval_idx = {s: torch.tensor([idx(r) for r in eval_rows[s]]) for s in eval_sets}
    eval_tracks = {s: [r["track_id"] for r in eval_rows[s]] for s in eval_sets}
    across = {s: torch.tensor(shuffled_across_index(eval_tracks[s])) for s in eval_sets}
    perms = {s: shuffled_time_perms(len(eval_rows[s]), t_muq, cfg["seed"] + k) for k, s in enumerate(eval_sets)}

    # Re-seed right before the translator exists: MuQ's forward pass consumes torch RNG, so without
    # this the initialization would depend on whether the feature cache was hit.
    seed_everything(cfg["seed"], device)
    model = MuQToMelTranslator(tensors["muq"].shape[-1], cfg["hidden_channels"], h.num_mels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["adam_betas"]))
    sampler = TrackBalancedSampler(pool_by_track, cfg["batch_size"], random.Random(f"task18-{cfg['seed']}-sampler"))
    resolved = {**cfg, "run_name": run_name, **hashes, "feature_cache": str(cache_path.relative_to(REPO_ROOT)),
                "num_pool_crops": len(pool_all), "n_train_tracks": len(pool_by_track), "eval_crops": {s: len(eval_rows[s]) for s in eval_sets},
                "translator_parameters": sum(p.numel() for p in model.parameters()),
                "receptive_field_mel_frames": model.receptive_field_mel_frames, "device_metadata": device_info,
                "seeding": seeding, "slurm": slurm_resources()}

    state = {"step": 0, "best": {"val_mel_l1": float("inf"), "step": 0}, "history": [], "wandb_run_id": None, "lineage": []}
    early_stopping = cfg.get("early_stopping")  # TASK-20: {patience, min_delta}; absent -> fixed step budget
    if resuming:
        ck = load_portable(latest)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        sampler.load_state_dict(ck["sampler_state"])
        torch.set_rng_state(ck["torch_cpu_rng"])
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        state = ck["train_state"]
        state.setdefault("early_stopping", None)
        print(f"resumed from {latest.relative_to(REPO_ROOT)} at step {state['step']}")
    if early_stopping and not state.get("early_stopping"):
        state["early_stopping"] = {"checks_since_improvement": 0, "stopped": False, "reason": None, "stopped_at_step": None}
    state["lineage"].append({"started_at_step": state["step"], "utc": datetime.now(UTC).isoformat(),
                             "slurm_job_id": os.environ.get("SLURM_JOB_ID")})

    from muq_inversion.wandb_logging import init_run, log_artifact_file

    run = init_run(config=resolved, environment=environment, run_name=run_name, mode=args.wandb_mode,
                   group=cfg["wandb_group"], job_type="heldout", tags=cfg["wandb_tags"], resume_id=state["wandb_run_id"])
    state["wandb_run_id"] = run.id
    run_manifest["wandb_run_id"] = run.id
    atomic_write_text(manifest_path, json.dumps(run_manifest, indent=2, default=str))

    @torch.inference_mode()
    def predict(m, eval_set: str, condition: str) -> torch.Tensor:
        """(n, num_mels, T) raw predicted mel on CPU."""
        ix = eval_idx[eval_set]
        if condition == "average_mel":
            return avg_mel[None, :, None].expand(len(ix), -1, t_mel).clone()
        x = conditioned_features(condition, feats_dev[ix.to(device)], across[eval_set].to(device), perms[eval_set])
        return torch.cat([m(x[i:i + 64], t_mel, positions).cpu() for i in range(0, len(ix), 64)])

    quick_sets = tuple(cfg.get("quick_eval_sets", ("memorized", "train_unseen", "val")))

    def quick_eval(sets=quick_sets) -> dict:
        model.eval()
        out = {}
        for s in sets:
            target = tensors["mel"][eval_idx[s]]
            for c in (CONDITIONS if s == "val" else ("correct",)):
                out[f"{s}/{c}/mel_l1"] = float((predict(model, s, c) - target).abs().mean())
        model.train()
        return out

    def checkpoint(path: Path, include_best_state: bool = False) -> None:
        ck = {"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
              "sampler_state": sampler.state_dict(), "torch_cpu_rng": torch.get_rng_state(),
              "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "train_state": state,
              "resolved_config": resolved, "device_metadata": device_info, "git_commit": environment.get("git_commit"),
              "git_dirty": environment.get("git_dirty"), "uv_lock_sha256": hashes["uv_lock_sha256"]}
        atomic_torch_save(path, ck)
        back = load_portable(path)  # validate immediately: metadata + a key tensor
        assert back["train_state"]["step"] == state["step"]
        k0 = next(iter(ck["model_state_dict"]))
        assert torch.equal(back["model_state_dict"][k0].cpu(), ck["model_state_dict"][k0].cpu())

    stop = {"flag": False}

    def on_signal(signum, _frame):
        print(f"signal {signum}: saving latest.pt after this step")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    # --- phase 2: training (skipped when evaluating existing checkpoints) ---
    lr_schedule = cfg.get("lr_schedule", {"type": "constant"})
    t0 = time.perf_counter()
    model.train()
    if evaluate_only:
        src_ckpt = REPO_ROOT / evaluate_only
        src_final = load_portable(src_ckpt / "final.pt")
        state["history"] = src_final["train_state"]["history"]
        sampler.load_state_dict(src_final["sampler_state"])
        run_manifest["evaluated_checkpoints"] = {n: {"path": str((src_ckpt / n).relative_to(REPO_ROOT)), "sha256": file_sha256(src_ckpt / n)}
                                                 for n in ("best.pt", "final.pt")}
        atomic_write_text(manifest_path, json.dumps(run_manifest, indent=2, default=str))
        for entry in state["history"]:  # replay the source run's curves into this arm's W&B run
            run.log({f"eval/{k}": v for k, v in entry.items() if "/" in k and not k.startswith("exposure")}
                    | {"train/l1_log_mel": entry["train_l1"]}, step=entry["step"])
        ckpt_dir = src_ckpt
        state["step"] = cfg["steps"]
    while state["step"] < cfg["steps"] and not (state.get("early_stopping") or {}).get("stopped"):
        step = state["step"] + 1
        batch = torch.tensor(sampler.next_batch(), device=device)
        with backend_stage(f"train step {step}", device):
            pred = model(feats_dev[batch], t_mel, positions)
            loss = (pred - mel_dev[batch]).abs().mean()
            if not torch.isfinite(loss):
                raise SystemExit(f"STOP CONDITION: non-finite loss at step {step}")
            opt.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))  # measure only
            lr_now = cfg["learning_rate"] * lr_multiplier(step, lr_schedule, cfg["steps"])
            for group in opt.param_groups:  # a pure function of the step, so resuming needs no extra state
                group["lr"] = lr_now
            opt.step()
        state["step"] = step
        log = {}
        if step % cfg["log_every"] == 0:
            log |= {"train/l1_log_mel": loss.item(), "train/grad_norm": grad_norm, "train/lr": lr_now}
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            ev = quick_eval()
            exposure = sampler.exposure_summary()
            improved = ev["val/correct/mel_l1"] < state["best"]["val_mel_l1"]
            if early_stopping:  # primary val, correct conditioning only -- the same metric as best.pt
                es = early_stopping_update(state["early_stopping"], ev["val/correct/mel_l1"], state["best"]["val_mel_l1"], early_stopping)
                if es["stopped"]:
                    es["stopped_at_step"] = step
                state["early_stopping"] = es
                log |= {"early_stopping/checks_since_improvement": es["checks_since_improvement"]}
            if improved:
                state["best"] = {"val_mel_l1": ev["val/correct/mel_l1"], "step": step}
                atomic_torch_save(ckpt_dir / "best.pt", {"model_state_dict": model.state_dict(), "step": step,
                                                         "val_mel_l1": ev["val/correct/mel_l1"], "resolved_config": resolved,
                                                         "device_metadata": device_info})
            state["history"].append({"step": step, "train_l1": loss.item(), **ev, **{f"exposure/{k}": v for k, v in exposure.items()}})
            log |= {f"eval/{k}": v for k, v in ev.items()} | {f"exposure/{k}": v for k, v in exposure.items()}
            log["eval/best_val_mel_l1"] = state["best"]["val_mel_l1"]
            print(f"step {step}: loss {loss.item():.4f} | mel L1 memorized {ev['memorized/correct/mel_l1']:.3f} "
                  f"train_unseen {ev['train_unseen/correct/mel_l1']:.3f} val {ev['val/correct/mel_l1']:.3f} "
                  f"(across {ev['val/shuffled_across/mel_l1']:.3f}, avg-mel {ev['val/average_mel/mel_l1']:.3f})"
                  f"{' *best*' if improved else ''} | tracks visited {exposure['tracks_visited']}/{exposure['tracks_total']}")
            checkpoint(latest)
            if step % cfg["checkpoint_every"] == 0:
                checkpoint(ckpt_dir / f"step_{step:07d}.pt")
        if log:
            run.log(log, step=step)
        if (state.get("early_stopping") or {}).get("stopped"):
            print(f"early stopping at step {step}: {state['early_stopping']['reason']} (best step {state['best']['step']})")
            break
        if stop["flag"] or step == args.stop_after_step:
            checkpoint(latest)
            run_manifest["starts"][-1]["interrupted_at_step"] = step
            atomic_write_text(manifest_path, json.dumps(run_manifest, indent=2, default=str))
            run.finish()
            print(f"interrupted at step {step}; resume with the same --run-name")
            return 3
    train_seconds = time.perf_counter() - t0
    if state.get("early_stopping") and not state["early_stopping"]["stopped"] and not evaluate_only:
        state["early_stopping"]["reason"] = f"max_steps: reached {cfg['steps']} steps"
        state["early_stopping"]["stopped_at_step"] = state["step"]
    if not evaluate_only and not (ckpt_dir / "final.pt").exists():
        checkpoint(ckpt_dir / "final.pt")

    # --- phase 3: evaluation ---
    stft = MultiResolutionSTFTLoss(*STFT)
    sr = h.sampling_rate

    @torch.inference_mode()
    def decode(mels: torch.Tensor) -> torch.Tensor:
        """(n, num_mels, T) raw mel -> (n, samples) audio, clamped to BigVGAN's floor first."""
        out = []
        with backend_stage("BigVGAN decode", device):
            for i in range(0, len(mels), cfg["decode_batch_size"]):
                out.append(bigvgan(mels[i:i + cfg["decode_batch_size"]].clamp(min=mel_floor).to(device)).squeeze(1).cpu())
        return torch.cat(out)

    print("oracle round trips ...")
    oracle = {}
    for s in eval_sets:
        target = tensors["mel"][eval_idx[s]]
        audio = decode(target)
        oracle[s] = {"audio": audio, "rt_mel_l1": [float((official_mel(audio[i:i + 1], h)[0] - target[i]).abs().mean())
                                                    for i in range(len(audio))]}

    def evaluate(state_dict: dict, label: str, with_audio: bool) -> tuple[dict, list[dict], dict]:
        m = MuQToMelTranslator(tensors["muq"].shape[-1], cfg["hidden_channels"], h.num_mels).to(device)
        m.load_state_dict(state_dict)
        m.eval()
        aggregates, examples, audio_out = {}, [], {}
        for s in eval_sets:
            target, wave = tensors["mel"][eval_idx[s]], tensors["wave"][eval_idx[s]]
            per_cond: dict[str, list[dict]] = {}
            for c in CONDITIONS:
                pred = predict(m, s, c)
                metrics = [mel_metrics(pred[i], target[i], mel_floor) for i in range(len(pred))]
                if with_audio:
                    audio = decode(pred)
                    audio_out[(s, c)] = (pred, audio)
                    for i, mt in enumerate(metrics):
                        mt |= contour_metrics(wave[i].numpy(), audio[i].numpy(), sr, cfg["onset_smoothing_ms"])
                        am = audio_metrics(wave[i], audio[i], h, stft)
                        mt |= {"resynthesis_mel_l1": am["resynthesis_mel_l1"], "stft_distance": am["stft_distance"],
                               "waveform_correlation_phase_dependent": am["waveform_correlation"]}
                per_cond[c] = metrics
            oracle_contours = []
            if with_audio:
                for i in range(len(wave)):
                    oc = contour_metrics(wave[i].numpy(), oracle[s]["audio"][i].numpy(), sr, cfg["onset_smoothing_ms"])
                    oracle_contours.append(oc)
            for i, r in enumerate(eval_rows[s]):
                for c in CONDITIONS:
                    examples.append({"checkpoint": label, "eval_set": s, "crop_id": r["crop_id"], "track_id": r["track_id"],
                                     "genre": r["genre"], "crop_start_seconds": r["crop_start_seconds"], "condition": c,
                                     "oracle_round_trip_mel_l1": oracle[s]["rt_mel_l1"][i], **per_cond[c][i]})
            metric_names = list(per_cond["correct"][0])
            agg = {"n_crops": len(eval_rows[s]), "n_tracks": len(set(eval_tracks[s])),
                   "oracle_round_trip_mel_l1": _summ(oracle[s]["rt_mel_l1"]),
                   "conditions": {c: {k: _summ([e[k] for e in per_cond[c]]) for k in metric_names} for c in CONDITIONS},
                   "paired_mel_l1": {c: cluster_bootstrap_delta([e["mel_l1"] for e in per_cond["correct"]],
                                                                [e["mel_l1"] for e in per_cond[c]], eval_tracks[s], True,
                                                                cfg["seed"], cfg["bootstrap_samples"])
                                     for c in CONDITIONS if c != "correct"}}
            if with_audio:
                agg["oracle_contours"] = {k: _summ([o[k] for o in oracle_contours]) for k in oracle_contours[0]}
                secondary = [k for k in metric_names if k in HIGHER_IS_BETTER or k.startswith("onset_corr")]
                secondary += ["resynthesis_mel_l1", "stft_distance"]
                agg["paired_secondary"] = {
                    k: {c: cluster_bootstrap_delta([e[k] for e in per_cond["correct"]], [e[k] for e in per_cond[c]],
                                                   eval_tracks[s], k in ("resynthesis_mel_l1", "stft_distance"),
                                                   cfg["seed"], cfg["bootstrap_samples"])
                        for c in CONDITIONS if c != "correct"} for k in secondary}
            aggregates[s] = agg
            corr = agg["conditions"]["correct"]["mel_l1"]["mean"]
            print(f"[{label}] {s:>12}: mel L1 correct {corr:.3f} | round trip {agg['oracle_round_trip_mel_l1']['mean']:.3f} | "
                  + " ".join(f"{c} {agg['conditions'][c]['mel_l1']['mean']:.3f}" for c in CONDITIONS if c != "correct"))
        return aggregates, examples, audio_out

    best_ck = load_portable(ckpt_dir / "best.pt")
    print(f"evaluating best checkpoint (step {best_ck['step']}, selected on val mel L1 only) ...")
    best_agg, best_examples, best_audio = evaluate(best_ck["model_state_dict"], "best", with_audio=True)
    final_agg, final_examples, _ = evaluate(load_portable(ckpt_dir / "final.pt")["model_state_dict"], "final", with_audio=False)
    outcome = classify_outcome(best_agg, cfg["outcome_rule"])
    final_outcome = classify_outcome(final_agg, cfg["outcome_rule"])

    # --- listening package (best checkpoint; crops fixed in the manifest) ---
    listen_dir = out_dir / "listening"
    listening = []
    for s in eval_sets:
        for i, r in enumerate(eval_rows[s]):
            if r["listening"] != "1":
                continue
            label = f"{s}_{r['track_id']}_{float(r['crop_start_seconds']):05.2f}s"
            d = listen_dir / s
            d.mkdir(parents=True, exist_ok=True)
            files = {"original": tensors["wave"][eval_idx[s][i]], "bigvgan_oracle": oracle[s]["audio"][i]}
            for c in ("correct", "shuffled_across", "shuffled_time", "constant_zero"):
                files[c] = best_audio[(s, c)][1][i]
            for k, v in files.items():
                sf.write(d / f"{label}_{k}.wav", v.numpy(), sr)
            fig, axes = plt.subplots(1, 4, figsize=(16, 3), sharey=True)
            panels = [("target", tensors["mel"][eval_idx[s][i]])] + [(c, best_audio[(s, c)][0][i]) for c in ("correct", "shuffled_across", "constant_zero")]
            for ax, (title, mel) in zip(axes, panels, strict=True):
                ax.imshow(mel.numpy(), origin="lower", aspect="auto", vmin=-11.5, vmax=2.5)
                ax.set_title(title)
            fig.suptitle(label)
            fig.tight_layout()
            fig.savefig(d / f"{label}_mel.png", dpi=90)
            plt.close(fig)
            ex = next(e for e in best_examples if e["crop_id"] == r["crop_id"] and e["condition"] == "correct")
            listening.append({"label": label, "eval_set": s, "track_id": r["track_id"], "genre": r["genre"],
                              "dir": str(d.relative_to(out_dir)), "mel_l1_correct": ex["mel_l1"],
                              "oracle_round_trip_mel_l1": ex["oracle_round_trip_mel_l1"]})

    # --- report ---
    examples = best_examples + final_examples
    with open(out_dir / "per_example.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for e in examples for k in e}))
        w.writeheader()
        w.writerows(examples)
    report = {
        "task": cfg["task"], "arm": cfg.get("arm"), "run_name": run_name, "wandb_run_id": run.id, "train_seconds_this_start": train_seconds,
        "best_checkpoint": {"step": best_ck["step"], "val_mel_l1": best_ck["val_mel_l1"], "selection": "lowest val mel L1 (correct conditioning); test never used"},
        "final_step": state["step"], "max_steps": cfg["steps"], "early_stopping": state.get("early_stopping"),
        "n_train_tracks": len(pool_by_track), "outcome_best": outcome, "outcome_final": final_outcome,
        "best": best_agg, "final": final_agg, "exposure": sampler.exposure_summary(),
        "track_exposure": sampler.track_exposure, "history": state["history"], "listening": listening,
        "lineage": state["lineage"], "resolved_config": resolved, "device_memory": memory_stats(device),
        "paths": {"run_dir": str(run_dir.relative_to(REPO_ROOT)), "artifacts": str(out_dir.relative_to(REPO_ROOT)),
                  "per_example_csv": "per_example.csv", "listening": "listening/"},
    }
    atomic_write_text(out_dir / "report.json", json.dumps(report, indent=2, default=_json_default))
    atomic_write_text(out_dir / cfg.get("results_name", "TASK-18_RESULTS.md"), render_results(report))
    run.summary.update({"outcome": outcome["code"], "best_step": best_ck["step"], "final_step": state["step"],
                        "n_train_tracks": len(pool_by_track),
                        "stopping_reason": (state.get("early_stopping") or {}).get("reason") or f"fixed budget {cfg['steps']} steps",
                        **{f"best/{s}/mel_l1_correct": best_agg[s]["conditions"]["correct"]["mel_l1"]["mean"] for s in eval_sets}})
    log_artifact_file(run, out_dir / "report.json", f"{cfg['task'].lower()}-report-{run_name}", "report")
    run.finish()
    run_manifest["finished_utc"] = datetime.now(UTC).isoformat()
    run_manifest["outcome"] = outcome["code"]
    atomic_write_text(manifest_path, json.dumps(run_manifest, indent=2, default=str))
    print(f"\nOUTCOME (best checkpoint, step {best_ck['step']}): {outcome['code']} -- {outcome['text']}")
    print(f"report: {(out_dir / 'report.json').relative_to(REPO_ROOT)}; listening: {listen_dir.relative_to(REPO_ROOT)}")
    return 0


def _summ(values) -> dict:
    a = np.asarray(values, float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std()), "n": int(a.size)}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


if __name__ == "__main__":
    raise SystemExit(main())
