"""TASK-24: fixed-crop decoder fitting ladder -- single entry point (tasks/TASK-24.md).

    uv run python scripts/task24_ladder.py all     --config configs/task24.yaml --device xpu --run-name task24_v1
    uv run python scripts/task24_ladder.py prepare --config configs/task24.yaml --device xpu --run-name task24_v1
    uv run python scripts/task24_ladder.py train   --config configs/task24.yaml --device xpu --run-name task24_v1 [--arms 1,4]
    uv run python scripts/task24_ladder.py report  --config configs/task24.yaml --device xpu --run-name task24_v1

prepare: deterministic nested selection (data/manifests/task24_ladder.{csv,json}, regenerated only if
  identical), fitting/monitoring tensors gathered from the TASK-22 first-block cache (no recomputation;
  the anchor row must equal TASK-23's cached pair bit for bit), a fresh MuQ/mel/waveform cross-check of
  the first crops, per-crop tensor hashes (runs/task24/<run>/ladder_pairs.pt), and the shared initial
  state = TASK-23 B's step-0 weights (init_state.pt).
train: each arm N, from init_state.pt, 200k batch-1 updates on shuffled passes, fixed LR schedule;
  resumable (latest.pt, sampler + RNG state); one writer per arm (lock file).
report: matched-step and matched-exposure tables, plots, spectrograms, BigVGAN audio, report.
Every stage is idempotent; a finished arm or report is never overwritten. SIGTERM/SIGINT: latest.pt
at the end of the current update, exit 3.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from muq_inversion.audio import load_resample_validate
from muq_inversion.bigvgan_common import (
    PEAK_NORMALIZE_TARGET,
    bigvgan_log_mel_floor,
    load_bigvgan,
    official_mel,
)
from muq_inversion.device import (
    DEVICE_CHOICES,
    device_metadata,
    require_fp32_default,
    seed_everything,
)
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.logging_utils import capture_environment_manifest, select_device
from muq_inversion.models.muq_mel_translator import MuQToMelTranslator
from muq_inversion.task18 import file_sha256
from muq_inversion.task22 import resolve_muq_layer
from muq_inversion.task23 import (
    atomic_torch_save,
    atomic_write_text,
    boundary_frames,
    load_pair,
    state_dict_sha256,
    tensor_sha256,
)
from muq_inversion.task24 import (
    ArmLock,
    LadderSpec,
    anchor_row_matches,
    lr_at,
    select_ladder,
    train_ladder_arm,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import itertools

from task23_overfit import regions, resolve_recipe, spectrogram_figure, write_wav
from train_task18_heldout import load_config

MANIFEST_FIELDS = ("rank", "role", "min_n", "showcase", "crop_id", "track_id", "genre", "path", "crop_start_sample",
                   "crop_start_seconds", "crop_num_samples")


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def arm_name(n: int) -> str:
    return f"N{n:04d}"


# --- prepare -----------------------------------------------------------------------------

def find_cache(cache_dir: Path) -> tuple[Path, dict]:
    hits = [p for p in sorted(cache_dir.glob("task18_v1_L1_*.pt"))]
    for p in hits:
        d = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        if d.get("key_parts", {}).get("muq_layer") == "conformer_block_1_output":
            return p, d
    raise SystemExit(f"STOP: no TASK-22 first-block cache in {cache_dir}")


def manifest_rows(fit: list[dict], monitor: list[dict], sizes: list[int], showcase: list[int], seed: int) -> list[dict]:
    out = []
    for rank, r in enumerate(fit):
        out.append({"rank": rank, "role": "anchor" if rank == 0 else "fit", "min_n": next(n for n in sorted(sizes) if rank < n),
                    "showcase": int(rank == 0 or rank in showcase), **{k: r[k] for k in MANIFEST_FIELDS[4:]}})
    for j, r in enumerate(monitor):
        out.append({"rank": j, "role": "monitor", "min_n": "", "showcase": 0, **{k: r[k] for k in MANIFEST_FIELDS[4:]}})
    return [{k: str(v) for k, v in row.items()} for row in out]


def write_or_verify_manifest(rows: list[dict], meta: dict, csv_path: Path, json_path: Path) -> str:
    text = ",".join(MANIFEST_FIELDS) + "\n" + "".join(",".join(r[k] for k in MANIFEST_FIELDS) + "\n" for r in rows)
    meta_text = json.dumps(meta, indent=2) + "\n"
    if csv_path.exists():
        if csv_path.read_text() != text or json.loads(json_path.read_text()) != meta:
            raise SystemExit(f"STOP: {csv_path} exists with a different selection; never overwritten")
        return "verified (identical)"
    atomic_write_text(csv_path, text)
    atomic_write_text(json_path, meta_text)
    return "written"


def build_selection(cfg: dict, run_dir: Path, available: set | None = None) -> tuple[list[dict], list[dict], dict]:
    """Nested fitting order + monitoring set from the committed manifests (and, for smoke only, the
    crops present in a cache); writes the ladder manifest, or verifies an existing one is identical."""
    dc = cfg["data"]
    rows = read_csv(REPO_ROOT / dc["crop_manifest"])
    tracks = {t["track_id"]: t for t in read_csv(REPO_ROOT / dc["track_manifest"])}
    pool = [r for r in rows if r["role"] == "train_pool" and r["split"] == "train" and tracks[r["track_id"]]["split"] == "train"]
    sizes = sorted(dc["sizes"])
    fit, monitor = select_ladder(pool, dc["anchor_crop_id"], sizes[-1], dc["monitor_size"], dc["selection_seed"], available=available)
    # checks: nesting is by construction (prefixes); disjointness and split
    fit_tracks, mon_tracks = {r["track_id"] for r in fit}, {r["track_id"] for r in monitor}
    if fit_tracks & mon_tracks or any(tracks[r["track_id"]]["split"] != "train" for r in fit + monitor):
        raise SystemExit("STOP: monitoring tracks overlap the fitting pool or a crop is not from a training track")
    if len({r["crop_id"] for r in fit}) != len(fit):
        raise SystemExit("STOP: duplicate fitting crop")
    showcase = [s for s in dc["showcase"] if s < len(fit)]
    mrows = manifest_rows(fit, monitor, sizes, showcase, dc["selection_seed"])
    genre_counts = {n: dict(sorted({g: sum(1 for r in fit[:n] if r["genre"] == g) for g in {r["genre"] for r in fit[:n]}}.items())) for n in sizes}
    mmeta = {"task": "TASK-24", "selection_seed": dc["selection_seed"], "sizes": sizes, "monitor_size": dc["monitor_size"],
             "anchor_crop_id": dc["anchor_crop_id"], "showcase_ranks": showcase,
             "algorithm": "muq_inversion.task24.select_ladder (genre round-robin over a seeded track order, one seeded crop per track; "
                          "anchor first; monitoring = next unused tracks)",
             "source_manifests": {k: file_sha256(REPO_ROOT / dc[k]) for k in ("crop_manifest", "track_manifest")},
             "distinct_tracks": {n: len({r["track_id"] for r in fit[:n]}) for n in sizes}, "genre_counts": genre_counts,
             "monitor_genres": dict(sorted({g: sum(1 for r in monitor if r["genre"] == g) for g in {r["genre"] for r in monitor}}.items())),
             "restricted_to_cache": available is not None}
    mmeta = json.loads(json.dumps(mmeta))  # the on-disk form (string keys), so verification compares like with like
    if dc["manifest_csv"]:
        status = write_or_verify_manifest(mrows, mmeta, REPO_ROOT / dc["manifest_csv"], REPO_ROOT / dc["manifest_json"])
    else:
        status = write_or_verify_manifest(mrows, mmeta, run_dir / "task24_ladder.csv", run_dir / "task24_ladder.json")
    print(f"ladder manifest {status}: {len(fit)} fitting crops ({mmeta['distinct_tracks']} distinct tracks), {len(monitor)} monitoring")
    return fit, monitor, mmeta


def prepare(cfg: dict, recipe: dict, device: torch.device, run_dir: Path, art: Path) -> dict:
    ladder_path, init_path = run_dir / "ladder_pairs.pt", run_dir / "init_state.pt"
    if ladder_path.exists() and init_path.exists():
        d = load_ladder(ladder_path)
        print(f"ladder pairs cache hit (hashes verified): {ladder_path.relative_to(REPO_ROOT)}")
        return d
    dc = cfg["data"]
    anchor_pair = load_pair(REPO_ROOT / dc["anchor_pair"])
    if anchor_pair["meta"]["crop"]["crop_id"] != dc["anchor_crop_id"]:
        raise SystemExit(f"STOP: {dc['anchor_pair']} is {anchor_pair['meta']['crop']['crop_id']}, not {dc['anchor_crop_id']}")
    cache_path, cache = find_cache(REPO_ROOT / dc["task22_cache_dir"])
    index = {k: i for i, k in enumerate(cache["keys"])}
    fit, monitor, mmeta = build_selection(cfg, run_dir, available=set(index) if dc["restrict_to_cache"] else None)

    # gather tensors from the cache (no recomputation); the anchor must be TASK-23's pair exactly
    A = anchor_pair["tensors"]

    def gather(rs: list[dict]) -> tuple[dict, list[dict]]:
        feats, mels, waves, info = [], [], [], []
        for r in rs:
            key = (r["path"], int(r["crop_start_sample"]))
            if r["crop_id"] == dc["anchor_crop_id"]:
                if key in index:
                    i = index[key]
                    if not anchor_row_matches(cache["tensors"], i, A):
                        raise SystemExit("STOP: the anchor's TASK-22 cache row differs from TASK-23's cached pair")
                    src = f"TASK-22 cache row {i} == TASK-23 pair.pt (bit-exact)"
                elif dc["anchor_must_be_in_cache"]:
                    raise SystemExit("STOP: anchor not in the TASK-22 cache")
                else:
                    src = "TASK-23 pair.pt (not in this cache; smoke)"
                f, m, w = A["features"][0], A["target_mel"][0], A["waveform"]
            else:
                if key not in index:
                    raise SystemExit(f"STOP: {r['crop_id']} not in the TASK-22 cache")
                i = index[key]
                f, m, w = (cache["tensors"][k][i].clone() for k in ("muq", "mel", "wave"))
                src = f"TASK-22 cache row {i}"
            if f.shape != A["features"][0].shape or m.shape != A["target_mel"][0].shape or w.shape != A["waveform"].shape:
                raise SystemExit(f"STOP: {r['crop_id']} tensor shapes differ from the anchor's")
            if not (torch.isfinite(f).all() and torch.isfinite(m).all() and torch.isfinite(w).all()):
                raise SystemExit(f"STOP: non-finite cached tensor for {r['crop_id']}")
            feats.append(f.float())
            mels.append(m.float())
            waves.append(w.float())
            info.append({"crop_id": r["crop_id"], "source": src, "features_sha256": tensor_sha256(f), "mel_sha256": tensor_sha256(m),
                         "wave_sha256": tensor_sha256(w)})
        return {"features": torch.stack(feats), "mel": torch.stack(mels), "wave": torch.stack(waves)}, info

    fit_t, fit_info = gather(fit)
    mon_t, mon_info = gather(monitor)
    cache = None  # release the memory map

    # fresh cross-check of the first crops: features (MuQ frozen, hook-checked), waveform, target mel
    bigvgan = load_bigvgan(device)
    h = bigvgan.h
    del bigvgan
    muq = FrozenMuQEncoder(recipe["muq_model_id"], device)
    layer = resolve_muq_layer(recipe["muq_layer"], len(muq.model.model.conformer.layers))
    with torch.inference_mode():
        ps0 = torch.stack([torch.stack([p.sum(), p.abs().sum()]) for p in muq.model.parameters()]).cpu()
    checks = []
    n_samples = recipe["num_mel_hops"] * recipe["hop_size"]
    for j, r in enumerate(fit[:dc["cross_check_crops"]]):
        track, _ = load_resample_validate(REPO_ROOT / r["path"], h.sampling_rate)
        track = librosa.util.normalize(track) * PEAK_NORMALIZE_TARGET
        s = int(r["crop_start_sample"])
        wave = torch.from_numpy(np.ascontiguousarray(track[s:s + n_samples])).float()
        mel = official_mel(wave[None], h)[0]
        fresh, chk = muq.encode_block_output(wave[None].to(device), layer["block"])
        c = {"crop_id": r["crop_id"], "wave_equal": bool(torch.equal(wave, fit_t["wave"][j])),
             "wave_max_abs_diff": float((wave - fit_t["wave"][j]).abs().max()),
             "mel_max_abs_diff": float((mel - fit_t["mel"][j]).abs().max()),
             "features_max_abs_diff": float((fresh.float().cpu()[0] - fit_t["features"][j]).abs().max()),
             "features_rel_rms_diff": float((fresh.float().cpu()[0] - fit_t["features"][j]).norm() / fit_t["features"][j].norm()),
             "index_check": chk}
        checks.append(c)
        if (c["wave_max_abs_diff"] > dc["wave_match_atol"] or c["mel_max_abs_diff"] > dc["mel_match_atol"]
                or c["features_rel_rms_diff"] > dc["feature_match_rel_rms"]):
            raise SystemExit(f"STOP: cache cross-check failed: {c}")
    with torch.inference_mode():
        ps1 = torch.stack([torch.stack([p.sum(), p.abs().sum()]) for p in muq.model.parameters()]).cpu()
    frozen = {"eval_mode": not muq.model.training, "all_requires_grad_false": all(not p.requires_grad for p in muq.model.parameters()),
              "parameters_unchanged": bool(torch.equal(ps0, ps1))}
    if not all(frozen.values()):
        raise SystemExit(f"STOP: MuQ not frozen: {frozen}")
    del muq

    # shared initial state: TASK-23 B's step-0 weights, also == a fresh TASK-22-seeded build
    ic = cfg["init"]
    src = torch.load(REPO_ROOT / ic["checkpoint"], map_location="cpu", weights_only=False)
    b_sha = json.loads((REPO_ROOT / ic["summary"]).read_text())["init_sha256"]
    seed_everything(recipe["seed"], torch.device("cpu"))
    model = MuQToMelTranslator(fit_t["features"].shape[-1], recipe["hidden_channels"], fit_t["mel"].shape[1])
    fresh_sha = state_dict_sha256(model)
    model.load_state_dict(src["model_state_dict"])
    init_sha = state_dict_sha256(model)
    if not (init_sha == b_sha == fresh_sha):
        raise SystemExit(f"STOP: initial state mismatch: checkpoint {init_sha[:12]}, TASK-23 B {b_sha[:12]}, fresh seed-{recipe['seed']} {fresh_sha[:12]}")
    atomic_torch_save(init_path, {"model_state_dict": model.state_dict(), "sha256": init_sha, "source": ic["checkpoint"]})

    cache_identity = {"path": str(cache_path.relative_to(REPO_ROOT)), "bytes": cache_path.stat().st_size,
                      "sha256": file_sha256(cache_path) if dc["hash_cache_file"] else None,
                      "key_parts": torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)["key_parts"]}
    meta = {"fit_crops": fit_info, "monitor_crops": mon_info, "manifest": mmeta, "cache": cache_identity, "cross_check": checks,
            "muq_frozen": frozen, "anchor_pair": {"path": dc["anchor_pair"], "sha256": anchor_pair["sha256"]},
            "init": {"sha256": init_sha, "source": ic["checkpoint"], "equals_task23_B": True, "equals_fresh_seed": recipe["seed"]},
            "sample_rate": h.sampling_rate}
    tensors = {"fit": fit_t, "monitor": mon_t, "mask": A["mask"].clone(), "positions": A["positions"].clone()}
    atomic_torch_save(ladder_path, {"tensors": tensors, "meta": meta, "rows": {"fit": fit, "monitor": monitor}})
    back = load_ladder(ladder_path)
    art.mkdir(parents=True, exist_ok=True)
    atomic_write_text(art / "ladder_meta.json", json.dumps(meta, indent=2, default=str))
    print(f"ladder pairs: fit {tuple(fit_t['features'].shape)}, monitor {tuple(mon_t['features'].shape)} | cross-check "
          f"{[(round(c['features_rel_rms_diff'], 5), c['mel_max_abs_diff'], c['wave_max_abs_diff']) for c in checks]} | init {init_sha[:12]} (= TASK-23 B)")
    return back


def load_ladder(path: Path) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    for group, infos in (("fit", d["meta"]["fit_crops"]), ("monitor", d["meta"]["monitor_crops"])):
        t = d["tensors"][group]
        for j, info in enumerate(infos):
            for k, key in (("features", "features_sha256"), ("mel", "mel_sha256"), ("wave", "wave_sha256")):
                if tensor_sha256(t[k][j]) != info[key]:
                    raise RuntimeError(f"{path}: {group} crop {info['crop_id']} {k} hash mismatch")
    return d


# --- train -------------------------------------------------------------------------------

def spec_for(cfg: dict, recipe: dict, n: int, showcase: list[int]) -> LadderSpec:
    t = cfg["training"]
    return LadderSpec(n=n, steps=t["steps"], base_lr=recipe["learning_rate"], betas=tuple(recipe["adam_betas"]),
                      lr_drops=tuple((int(a), float(f)) for a, f in t["lr_drops"]), milestones=tuple(t["milestones"]),
                      exposures=tuple(t["exposures"]), log_every=t["log_every"], fit_eval_every=t["fit_eval_every"],
                      monitor_eval_every=t["monitor_eval_every"], latest_every=t["latest_every"], chunk=t["eval_chunk"],
                      sampler_seed=f"task24-sampler-{cfg['data']['selection_seed']}-N{n}", spike_factor=t["spike_factor"],
                      recovery_factor=t["recovery_factor"], anchor_index=0, showcase=tuple([0] + [s for s in showcase if s < n]))


def train_arms(cfg: dict, recipe: dict, device: torch.device, run_dir: Path, ladder: dict, arms: list[int],
               stop_after: tuple[int, int] | None, steal_lock: bool, stop: dict) -> int:
    init = torch.load(run_dir / "init_state.pt", map_location="cpu", weights_only=False)
    T = ladder["tensors"]
    mask, positions = T["mask"], T["positions"].to(device)
    t_mel, t_muq = T["fit"]["mel"].shape[-1], T["fit"]["features"].shape[1]
    showcase = ladder["meta"]["manifest"]["showcase_ranks"]
    for n in arms:
        model = MuQToMelTranslator(T["fit"]["features"].shape[-1], recipe["hidden_channels"], T["fit"]["mel"].shape[1])
        model.load_state_dict(init["model_state_dict"])
        if state_dict_sha256(model) != init["sha256"]:
            raise SystemExit(f"STOP: N={n} initial weights differ from init_state.pt")
        model = model.to(device)
        boundary = boundary_frames(t_mel, (model.receptive_field_mel_frames - 1) // 2, T["positions"], t_muq)
        data = {"fit_feats": T["fit"]["features"][:n].to(device), "fit_mel": T["fit"]["mel"][:n].to(device),
                "mon_feats": T["monitor"]["features"].to(device), "mon_mel": T["monitor"]["mel"].to(device),
                "mask": mask, "boundary": boundary, "positions": positions}
        spec = spec_for(cfg, recipe, n, showcase)
        arm_dir = run_dir / arm_name(n)
        lock = ArmLock(arm_dir / "LOCK", steal=steal_lock)
        try:
            print(f"=== arm N={n}: {spec.steps} updates, schedule {spec.lr_drops}, exposure steps {spec.exposure_steps()}, "
                  f"init {init['sha256'][:12]}")
            summ = train_ladder_arm(spec, model, data, arm_dir, device,
                                    meta={"init_sha256": init["sha256"], "fit_crop_ids": [c["crop_id"] for c in ladder["meta"]["fit_crops"][:n]]},
                                    stop_after_step=stop_after[1] if stop_after and stop_after[0] == n else None,
                                    should_stop=lambda: stop["flag"])
        finally:
            lock.release()
        if summ is None:
            return 3
        print(f"[N={n}] done: final fit L1 {summ['final']['l1']:.5f} (anchor {summ['final']['anchor_l1']:.5f}), "
              f"best {summ['best']['fit_l1']:.5f} @ {summ['best']['step']}, {summ['runtime_seconds'] / 60:.1f} min")
        if stop["flag"]:
            return 3
    return 0


# --- report ------------------------------------------------------------------------------

def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def report(cfg: dict, recipe: dict, device: torch.device, run_dir: Path, art: Path, ladder: dict, env: dict, dev_info: dict) -> None:
    sizes = sorted(cfg["data"]["sizes"])
    S = {n: json.loads((run_dir / arm_name(n) / "summary.json").read_text()) for n in sizes}
    fe = {n: jl(run_dir / arm_name(n) / "fit_eval.jsonl") for n in sizes}
    me = {n: jl(run_dir / arm_name(n) / "monitor_eval.jsonl") for n in sizes}
    tl = {n: jl(run_dir / arm_name(n) / "train_log.jsonl") for n in sizes}
    t = cfg["training"]
    drops = [a for a, _ in t["lr_drops"]]
    art.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sizes)))

    def annotate(ax, n, c):
        for d in drops:
            ax.axvline(d, color="0.6", ls=":", lw=1)
        for sp in S[n]["spikes"]:
            ax.axvline(sp["step"], color=c, ls="--", lw=0.6, alpha=0.6)

    # 1. fitting-set L1 vs steps, and train-loss window max (spikes)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    for n, c in zip(sizes, colors, strict=True):
        r = [x for x in fe[n] if x["step"] > 0]
        ax[0].plot([x["step"] for x in r], [x["l1"] for x in r], color=c, lw=1, label=f"N={n}")
        ax[0].plot([x["step"] for x in r], [x["per_crop_max"] for x in r], color=c, lw=0.6, ls="--")
        ax[1].plot([x["step"] for x in tl[n]], [x["train_loss_window_max"] for x in tl[n]], color=c, lw=0.5, label=f"N={n}")
        annotate(ax[0], n, c)
    for a, title in ((ax[0], "whole fitting set, direct mel L1 (solid: aggregate, dashed: worst crop)"),
                     (ax[1], "train loss, max per 100-update window (spikes)")):
        a.set_yscale("log")
        a.set_xlabel("optimizer update (dotted grey: LR x0.1; dashed: spike events)")
        a.set_title(title)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(art / "fit_l1_vs_steps.png", dpi=110)
    plt.close(fig)

    # 2. fitting-set L1 vs presentations per crop
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n, c in zip(sizes, colors, strict=True):
        r = [x for x in fe[n] if x["step"] > 0]
        ax.plot([x["presentations_per_crop"] for x in r], [x["l1"] for x in r], color=c, lw=1, label=f"N={n}")
    for k in t["exposures"]:
        ax.axvline(k, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("presentations per crop (updates / N); dotted: matched-exposure points")
    ax.set_ylabel("whole fitting set mel L1")
    ax.set_title("Fitting L1 vs exposure (different N reach a given exposure at different updates and LR phases)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(art / "fit_l1_vs_presentations.png", dpi=110)
    plt.close(fig)

    # 3. monitoring set, separately
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n, c in zip(sizes, colors, strict=True):
        ax.plot([x["step"] for x in me[n]], [x["l1"] for x in me[n]], color=c, lw=1, marker=".", ms=3, label=f"N={n}")
        annotate(ax, n, c)
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("monitoring set mel L1")
    ax.set_title("MONITORING set: 16 crops of training tracks outside the 256 pool (never used for selection or stopping)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(art / "monitor_l1_vs_steps.png", dpi=110)
    plt.close(fig)

    # 4. per-crop distributions vs N, 5. anchor vs N
    def at_step(n, s):
        return next((x for x in fe[n] if x["step"] == s), None)

    def exposure_getter(k):
        return lambda n: at_step(n, k * n) if k * n <= t["steps"] else None

    def step_getter(s):
        return lambda n: at_step(n, s)

    points = [("final (200k)" if t["steps"] == 200000 else f"final ({t['steps']})", lambda n: at_step(n, t["steps"])),
              ("best", lambda n: {"per_crop_l1": S[n]["best"]["per_crop_l1"], "anchor_l1": S[n]["best"]["anchor_l1"]})]
    points += [(f"{k}x exposure", exposure_getter(k)) for k in t["exposures"]]
    fig, ax = plt.subplots(1, len(points), figsize=(4.5 * len(points), 4.5), sharey=True)
    for a, (name, get) in zip(ax, points, strict=True):
        data = [(n, get(n)) for n in sizes]
        data = [(n, r) for n, r in data if r is not None]
        a.boxplot([r["per_crop_l1"] for _, r in data], tick_labels=[str(n) for n, _ in data], whis=(0, 100))
        for k, (n, r) in enumerate(data, 1):
            a.scatter(np.full(len(r["per_crop_l1"]), k) + np.random.default_rng(0).uniform(-0.15, 0.15, len(r["per_crop_l1"])),
                      r["per_crop_l1"], s=4, alpha=0.4, color="C0")
        a.set_yscale("log")
        a.set_title(name)
        a.set_xlabel("N")
    ax[0].set_ylabel("per-crop direct mel L1 (whiskers: min-max)")
    fig.suptitle("Per-crop fitting error distribution vs N")
    fig.tight_layout()
    fig.savefig(art / "per_crop_distribution_vs_n.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, get in points + [(f"step {s}", step_getter(s)) for s in t["milestones"][:1]]:
        xs = [n for n in sizes if get(n) is not None]
        ax.plot(xs, [get(n)["anchor_l1"] for n in xs], marker="o", label=name)
    ax.axhline(cfg["report"]["task23_anchor_best_l1"], color="k", ls=":", lw=1, label="TASK-23 B best (different schedule/budget)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (fitting crops)")
    ax.set_ylabel("anchor-crop mel L1")
    ax.set_title("TASK-23 anchor crop error vs N")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(art / "anchor_l1_vs_n.png", dpi=110)
    plt.close(fig)

    # spectrograms + audio for the anchor and the pre-selected showcase crops
    bigvgan = load_bigvgan(device)
    h = bigvgan.h
    floor = bigvgan_log_mel_floor()
    T = ladder["tensors"]["fit"]
    rows = ladder["rows"]["fit"]
    showcase = [0] + ladder["meta"]["manifest"]["showcase_ranks"]
    rc = cfg["report"]
    audio_dir = art / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio = {"reference": {}, "arms": {}}

    def vocode(mel: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return bigvgan(mel.clamp(min=floor)[None].to(device)).float().cpu()[0, 0]

    reg = {}
    for rank in showcase:
        cid = rows[rank]["crop_id"].replace(":", "_")
        reg[rank] = regions(T["mel"][rank][None], T["wave"][rank].numpy(), h.sampling_rate, h.hop_size, rc["zoom_frames"], rc["harmonic_max_bin"])
        audio["reference"][rank] = {"crop_id": rows[rank]["crop_id"],
                                    "original": write_wav(audio_dir / f"r{rank}_{cid}_original.wav", T["wave"][rank], h.sampling_rate),
                                    "ground_truth_mel": write_wav(audio_dir / f"r{rank}_{cid}_ground_truth_mel_bigvgan.wav", vocode(T["mel"][rank]), h.sampling_rate)}
    for n in sizes:
        d = art / arm_name(n)
        d.mkdir(parents=True, exist_ok=True)
        pr = run_dir / arm_name(n) / "preds"
        best, final = (torch.load(pr / f, map_location="cpu", weights_only=False) for f in ("best.pt", "final.pt"))
        audio["arms"][n] = {}
        for j, rank in enumerate(best["showcase"]):
            cid = rows[rank]["crop_id"].replace(":", "_")
            for label, pred, step in (("best", best["fit"][j], best["step"]), ("final", final["fit"][rank], final["step"])):
                tgt = T["mel"][rank].numpy()
                spectrogram_figure(tgt, pred.numpy(), f"N={n}, rank {rank} ({rows[rank]['crop_id']}), {label} (step {step})", d / f"r{rank}_{label}_spectrogram.png")
                spectrogram_figure(tgt, pred.numpy(), f"N={n}, rank {rank}, {label}: transient (frame {reg[rank]['transient_frame']})",
                                   d / f"r{rank}_{label}_zoom_transient.png", frames=tuple(reg[rank]["transient_window"]))
                spectrogram_figure(tgt, pred.numpy(), f"N={n}, rank {rank}, {label}: harmonic region",
                                   d / f"r{rank}_{label}_zoom_harmonic.png", frames=tuple(reg[rank]["harmonic_window"]), bins=tuple(reg[rank]["harmonic_bins"]))
                if label == "final" and best["step"] == final["step"]:
                    continue
                a = write_wav(d / f"r{rank}_{cid}_{label}_pred_mel_bigvgan.wav", vocode(pred), h.sampling_rate)
                audio["arms"][n][f"r{rank}_{label}"] = a | {"step": step, "direct_mel_l1": float((pred - T["mel"][rank]).abs().mean())}

    # tables
    def brief(r):
        return None if r is None else {k: r[k] for k in ("l1", "per_crop_median", "per_crop_p90", "per_crop_max", "anchor_l1", "l1_interior",
                                                         "l1_boundary", "time_diff_l1", "freq_diff_l1") if k in r}
    mon = {n: {x["step"]: x for x in me[n]} for n in sizes}
    tables = {"matched_steps": {s: {n: {"fit": brief(at_step(n, s)), "monitor_l1": (mon[n].get(s) or {}).get("l1"), "lr": lr_at(s, recipe["learning_rate"], tuple(map(tuple, t["lr_drops"])))}
                                    for n in sizes} for s in t["milestones"]},
              "matched_exposure": {k: {n: ({"step": k * n, "lr": lr_at(k * n, recipe["learning_rate"], tuple(map(tuple, t["lr_drops"]))),
                                            "fit": brief(at_step(n, k * n)), "monitor_l1": (mon[n].get(k * n) or {}).get("l1")} if k * n <= t["steps"] else None)
                                       for n in sizes} for k in t["exposures"]},
              "best": {n: {"step": S[n]["best"]["step"], "fit_l1": S[n]["best"]["fit_l1"], "anchor_l1": S[n]["best"]["anchor_l1"]} for n in sizes}}
    final_step = t["steps"]
    obs = observations(sizes, tables, S, fe, drops, final_step)
    rep = {"task": "TASK-24", "run_name": run_dir.name, "finished_utc": datetime.now(UTC).isoformat(), "device": dev_info, "environment": env,
           "config": cfg, "ladder_meta": {k: v for k, v in ladder["meta"].items() if k not in ("fit_crops", "monitor_crops")},
           "summaries": {n: {k: v for k, v in s.items() if k not in ("presentation_counts", "best")} | {"best_step": s["best"]["step"], "best_fit_l1": s["best"]["fit_l1"],
                                                                                                     "presentations_min_max": [min(s["presentation_counts"]), max(s["presentation_counts"])]}
                         for n, s in S.items()},
           "tables": tables, "observations": obs, "audio": audio, "listening": "pending"}
    atomic_write_text(art / "report.json", json.dumps(rep, indent=2, default=str))
    write_markdown(art, rep, sizes, cfg)
    print(f"report: {(art / 'TASK-24_RESULTS.md').relative_to(REPO_ROOT)}")


def observations(sizes, tables, S, fe, drops, final_step) -> dict:
    fin = {n: tables["matched_steps"][final_step][n]["fit"] for n in sizes}
    l1 = [fin[n]["l1"] for n in sizes]
    anchor = [fin[n]["anchor_l1"] for n in sizes]
    conc = {n: {"max_over_median": fin[n]["per_crop_max"] / fin[n]["per_crop_median"], "p90_over_median": fin[n]["per_crop_p90"] / fin[n]["per_crop_median"]}
            for n in sizes}

    def at(n, s):
        return next((x["l1"] for x in fe[n] if x["step"] == s), None)

    lr_effect = {n: {f"after_drop_at_{d}": {"before": at(n, d), "at_next_milestone": at(n, nxt), "ratio": (at(n, nxt) / at(n, d)) if at(n, d) and at(n, nxt) else None}
                     for d, nxt in zip(drops, drops[1:] + [final_step], strict=True)} for n in sizes}
    mon = {n: tables["matched_steps"][final_step][n]["monitor_l1"] for n in sizes}
    return {"final_fit_l1_by_n": dict(zip(sizes, l1, strict=True)), "final_fit_l1_monotone_increasing": all(a < b for a, b in itertools.pairwise(l1)),
            "final_anchor_l1_by_n": dict(zip(sizes, anchor, strict=True)), "anchor_monotone_increasing": all(a < b for a, b in itertools.pairwise(anchor)),
            "concentration_final": conc, "lr_reduction_effect": lr_effect, "final_monitor_l1_by_n": mon,
            "spike_counts": {n: len(S[n]["spikes"]) for n in sizes}}


def write_markdown(art: Path, rep: dict, sizes: list[int], cfg: dict) -> None:
    t, T, O = cfg["training"], rep["tables"], rep["observations"]
    L = [f"# TASK-24 results: fixed-crop decoder fitting ladder (`{rep['run_name']}`)", "",
         (f"Executed on {rep['device']['device_name']}, commit {rep['environment'].get('git_commit')} "
         f"(dirty={rep['environment'].get('git_dirty')}), finished {rep['finished_utc']}. All errors are direct pre-vocoder mel L1 "
         "(natural-log mel units), evaluation mode, aggregated over every valid mel element of the set."), "",
         (f"Init: TASK-23 B's step-0 weights (`{rep['ladder_meta']['init']['sha256'][:16]}`), identical for every arm. Schedule: "
         f"{t['steps']:,} batch-1 updates, LR 1e-3 x0.1 after {t['lr_drops'][0][0]:,} and after {t['lr_drops'][1][0]:,}. "
         "Comparisons with TASK-23 (20k updates, constant LR) are not pure dataset-size comparisons."), "",
         "## Matched optimizer steps (fitting set: aggregate / median / p90 / max per-crop; anchor; monitoring)", "",
         "| step | LR | " + " | ".join(f"N={n}" for n in sizes) + " |", "| --- | --- | " + " | ".join("---" for _ in sizes) + " |"]
    for s, row in T["matched_steps"].items():
        cells = []
        for n in sizes:
            f = row[n]["fit"]
            cells.append("-" if f is None else f"{f['l1']:.4f} / {f['per_crop_median']:.4f} / {f['per_crop_p90']:.4f} / {f['per_crop_max']:.4f}; "
                                               f"anchor {f['anchor_l1']:.4f}; mon {row[n]['monitor_l1']:.3f}")
        L.append(f"| {int(s):,} | {row[sizes[0]]['lr']:g} | " + " | ".join(cells) + " |")
    L += ["", "## Matched exposure (presentations per crop; reached after exactly k x N updates, so in different LR phases)", "",
          "| exposure | " + " | ".join(f"N={n}" for n in sizes) + " |", "| --- | " + " | ".join("---" for _ in sizes) + " |"]
    for k, row in T["matched_exposure"].items():
        cells = ["not reached" if row[n] is None else
                 f"step {row[n]['step']:,} (LR {row[n]['lr']:g}): {row[n]['fit']['l1']:.4f} / max {row[n]['fit']['per_crop_max']:.4f}; anchor "
                 f"{row[n]['fit']['anchor_l1']:.4f}; mon {row[n]['monitor_l1']:.3f}" for n in sizes]
        L.append(f"| {k}x | " + " | ".join(cells) + " |")
    L += ["", "## Best checkpoint (lowest whole-fitting-set L1; the monitoring set never selects)", "",
          "| N | step | fit L1 | anchor L1 | spikes | presentations min-max | runtime | updates/s |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n in sizes:
        s = rep["summaries"][n]
        b = T["best"][n]
        L.append(f"| {n} | {b['step']:,} | {b['fit_l1']:.5f} | {b['anchor_l1']:.5f} | {len(s['spikes'])} | {s['presentations_min_max'][0]:,}-{s['presentations_min_max'][1]:,} | "
                 f"{s['runtime_seconds'] / 60:.1f} min | {s['updates_per_second_median']:.0f} |")
    L += ["", f"TASK-23 B best on the anchor: {cfg['report']['task23_anchor_best_l1']} (20k updates, constant LR).", "",
          "## Observations (computed)", "",
          f"- Final whole-set L1 by N: {', '.join(f'{n}: {v:.4f}' for n, v in O['final_fit_l1_by_n'].items())}; strictly increasing with N: {O['final_fit_l1_monotone_increasing']}.",
          f"- Final anchor L1 by N: {', '.join(f'{n}: {v:.4f}' for n, v in O['final_anchor_l1_by_n'].items())}; strictly increasing: {O['anchor_monotone_increasing']}.",
          "- Concentration at the final step (max/median, p90/median per-crop L1): "
          + ", ".join(f"N={n}: {c['max_over_median']:.2f}, {c['p90_over_median']:.2f}" for n, c in O["concentration_final"].items()) + ".",
          "- LR reductions (whole-set L1 just before each drop vs the next milestone): "
          + "; ".join(f"N={n}: " + ", ".join(f"{k.split('_')[-1]}: {v['before']:.4f} -> {v['at_next_milestone']:.4f}" for k, v in e.items() if v["ratio"])
                      for n, e in O["lr_reduction_effect"].items()) + ".",
          f"- Final monitoring L1 by N: {', '.join(f'{n}: {v:.3f}' for n, v in O['final_monitor_l1_by_n'].items())} (transfer to unseen training tracks).",
          f"- Spike events (whole-set L1 > {t['spike_factor']}x best so far): {O['spike_counts']}; causes not attributed.", "",
          "## Interpretation limits", "",
          "- Low training error with poor monitoring performance suggests a transfer problem, not necessarily insufficient data alone.",
          "- Increasing training error with N motivates capacity/architecture and optimization ablations; it does not identify which is responsible.",
          "- A plateau does not establish information loss. Successful fitting does not establish invertibility or correct alignment.",
          "- Matched steps give different exposure per crop; matched exposure falls at different update counts and LR phases. Neither isolates every factor.",
          "- One nested sample sequence and one initialization: not a seed-robust scaling law.", "",
          "## Files", "", ("Plots: `fit_l1_vs_steps.png`, `fit_l1_vs_presentations.png`, `monitor_l1_vs_steps.png`, `per_crop_distribution_vs_n.png`, "
          "`anchor_l1_vs_n.png`; per arm `N0001/`..: spectrograms and zooms (anchor + showcase), BigVGAN WAVs (float, gain 1.0); references in `audio/`. "
          "Machine-readable: `report.json`, `runs/task24/<run>/N*/{fit_eval,monitor_eval,train_log}.jsonl`, `summary.json`. Listening: **pending**.")]
    atomic_write_text(art / "TASK-24_RESULTS.md", "\n".join(L) + "\n")


# --- main --------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["manifest", "prepare", "train", "report", "all"],
                        help="manifest: write/verify the tracked ladder manifest only (no cache or accelerator needed)")
    parser.add_argument("--config", type=Path, default=Path("configs/task24.yaml"))
    parser.add_argument("--device", choices=DEVICE_CHOICES, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--arms", default=None, help="comma-separated N values (default: all sizes)")
    parser.add_argument("--stop-after", default=None, help="N:STEP -- simulate an interruption (resume test); exit 3")
    parser.add_argument("--steal-lock", action="store_true", help="remove a stale arm LOCK (only if no job is running that arm)")
    parser.add_argument("--deterministic", action="store_true", help="torch.use_deterministic_algorithms(True) (CPU resume checks)")
    args = parser.parse_args()
    if args.stage == "manifest":
        cfg = load_config(REPO_ROOT / args.config)
        if cfg["data"]["restrict_to_cache"] or not cfg["data"]["manifest_csv"]:
            raise SystemExit("the manifest stage is for the full config (tracked manifest, no cache restriction)")
        build_selection(cfg, REPO_ROOT / cfg["run_root"])
        return 0
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    cfg = load_config(REPO_ROOT / args.config)
    recipe = resolve_recipe(cfg)
    run_dir, art = REPO_ROOT / cfg["run_root"] / args.run_name, REPO_ROOT / cfg["artifact_root"] / args.run_name
    if args.stage in ("report", "all") and (art / "report.json").exists():
        raise SystemExit(f"{art / 'report.json'} exists: the report of {args.run_name} is never overwritten")
    device = select_device(args.device or cfg["device_request"], cfg["allow_cpu_fallback"])
    require_fp32_default()
    dev_info = device_metadata(args.device or cfg["device_request"], device)
    env = capture_environment_manifest(REPO_ROOT)
    print(f"TASK-24 {args.stage} {args.run_name} | device {json.dumps(dev_info)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    mpath = run_dir / "run_manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"task": "TASK-24", "run_name": args.run_name, "config": cfg, "recipe": recipe,
                                                                 "uv_lock_sha256": file_sha256(REPO_ROOT / "uv.lock"), "starts": []}
    if man["config"] != cfg:
        raise SystemExit(f"STOP: {args.config} differs from the config this run started with")
    man["starts"].append({"utc": datetime.now(UTC).isoformat(), "stage": args.stage, "arms": args.arms, "device": dev_info, "environment": env,
                          "argv": sys.argv, "deterministic_algorithms": args.deterministic})
    atomic_write_text(mpath, json.dumps(man, indent=2, default=str))
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    if args.stage in ("prepare", "all"):
        ladder = prepare(cfg, recipe, device, run_dir, art)
    else:
        if not (run_dir / "ladder_pairs.pt").exists():
            raise SystemExit(f"{run_dir / 'ladder_pairs.pt'} missing: run the prepare stage first")
        ladder = load_ladder(run_dir / "ladder_pairs.pt")
    if args.stage in ("train", "all"):
        arms = [int(x) for x in args.arms.split(",")] if args.arms else sorted(cfg["data"]["sizes"])
        bad = set(arms) - set(cfg["data"]["sizes"])
        if bad:
            raise SystemExit(f"unknown arms {sorted(bad)}")
        sa = tuple(int(x) for x in args.stop_after.split(":")) if args.stop_after else None
        code = train_arms(cfg, recipe, device, run_dir, ladder, arms, sa, args.steal_lock, stop)
        if code:
            return code
    if args.stage in ("report", "all"):
        missing = [n for n in cfg["data"]["sizes"] if not (run_dir / arm_name(n) / "summary.json").exists()]
        if missing:
            raise SystemExit(f"arms not finished: {missing}")
        report(cfg, recipe, device, run_dir, art, ladder, env, dev_info)
    return 0


if __name__ == "__main__":
    t0 = time.time()
    code = main()
    print(f"elapsed {time.time() - t0:.0f} s")
    raise SystemExit(code)
