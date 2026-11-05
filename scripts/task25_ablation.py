"""TASK-25: architecture-versus-optimization ablation on TASK-24's N=256 pool -- single runner (tasks/TASK-25.md).

    uv run python scripts/task25_ablation.py all     --config configs/task25.yaml --device mps --run-name task25_v1
    uv run python scripts/task25_ablation.py prepare --config configs/task25.yaml --device mps --run-name task25_v1
    uv run python scripts/task25_ablation.py train   --config configs/task25.yaml --device mps --run-name task25_v1 --arms A
    uv run python scripts/task25_ablation.py report  --config configs/task25.yaml --device mps --run-name task25_v1

prepare: verifies TASK-24's cached pairs (per-crop hashes), the manifest identity, monitoring disjointness,
  the TASK-23 anchor (bit-exact), and the shared initial state; records architecture sizes. No recomputation.
train: each arm for exactly 819,200 crop presentations (batch 1 or 16), exposure-indexed LR schedule,
  resumable, one writer per arm (lock), one W&B run per arm (online, offline fallback; JSONL mirror).
  Different arms may run in parallel processes.
report: primary fixed-exposure comparison, matched-update comparison, paired per-crop differences, plots,
  spectrograms, BigVGAN audio, report.json, TASK-25_RESULTS.md. Never overwrites a finished report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from muq_inversion.bigvgan_common import bigvgan_log_mel_floor, load_bigvgan
from muq_inversion.device import (
    DEVICE_CHOICES,
    device_metadata,
    require_fp32_default,
    seed_everything,
)
from muq_inversion.logging_utils import capture_environment_manifest, select_device
from muq_inversion.models.muq_mel_translator import MuQToMelTranslator
from muq_inversion.task18 import file_sha256
from muq_inversion.task23 import (
    atomic_write_text,
    boundary_frames,
    load_pair,
    state_dict_sha256,
)
from muq_inversion.task24 import ArmLock
from muq_inversion.task25 import (
    ExposureSpec,
    build_model,
    forward_macs,
    paired_difference,
    source_receptive_field,
    train_exposure_arm,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from task23_overfit import regions, resolve_recipe, spectrogram_figure, write_wav
from task24_ladder import load_ladder
from train_task18_heldout import load_config

COLORS = {"A": "C0", "B": "C1", "C": "C2", "D": "C3", "E": "C4", "F": "C5"}


# --- prepare -----------------------------------------------------------------------------

def load_inputs(cfg: dict) -> tuple[dict, dict]:
    dc = cfg["data"]
    ladder = load_ladder(REPO_ROOT / dc["ladder_pairs"])  # per-crop hashes verified
    init = torch.load(REPO_ROOT / dc["init_state"], map_location="cpu", weights_only=False)
    return ladder, init


def make_model(cfg: dict, recipe: dict, arm: dict, init: dict, n_in: int, n_mels: int) -> torch.nn.Module:
    """A-D: the shared initial state. E: seed_everything(init_seed), then a 2x-wide build. F: the same seed
    and build order, then the shared initial state copied into the trunk (resampler starts at linear)."""
    seed_everything(cfg["init_seed"], torch.device("cpu"))
    m = build_model(arm["arch"], n_in, recipe["hidden_channels"], n_mels, arm.get("width_multiplier", 1), arm.get("resampler_hidden", 32))
    if arm["init"] == "shared":
        m.load_state_dict(init["model_state_dict"])
    elif arm["init"] == "shared_trunk":
        res = m.load_state_dict(init["model_state_dict"], strict=False)
        if res.unexpected_keys or any(not k.startswith("resampler.") for k in res.missing_keys):
            raise SystemExit(f"STOP: trunk copy into F does not match: {res}")
    elif arm["init"] != "seed":
        raise SystemExit(f"unknown init {arm['init']}")
    return m


def prepare(cfg: dict, recipe: dict, run_dir: Path) -> dict:
    out = run_dir / "prepared.json"
    if out.exists():
        print(f"prepared (cached): {out.relative_to(REPO_ROOT)}")
        return json.loads(out.read_text())
    dc = cfg["data"]
    ladder, init = load_inputs(cfg)
    T, meta = ladder["tensors"], ladder["meta"]
    with open(REPO_ROOT / dc["manifest_csv"], newline="") as f:
        man = list(csv.DictReader(f))
    fit_ids = [r["crop_id"] for r in man if r["role"] in ("anchor", "fit")]
    mon_ids = [r["crop_id"] for r in man if r["role"] == "monitor"]
    if [c["crop_id"] for c in meta["fit_crops"]] != fit_ids or [c["crop_id"] for c in meta["monitor_crops"]] != mon_ids:
        raise SystemExit("STOP: TASK-24 cached pairs do not match data/manifests/task24_ladder.csv")
    if len(fit_ids) != dc["n_fit"] or T["fit"]["features"].shape[0] != dc["n_fit"]:
        raise SystemExit("STOP: fitting pool is not N=256")
    fit_tracks = {r["track_id"] for r in man if r["role"] in ("anchor", "fit")}
    mon_tracks = {r["track_id"] for r in man if r["role"] == "monitor"}
    if fit_tracks & mon_tracks:
        raise SystemExit("STOP: monitoring tracks overlap the fitting pool")
    A = load_pair(REPO_ROOT / dc["anchor_pair"])
    anchor_ok = (torch.equal(T["fit"]["features"][0], A["tensors"]["features"][0]) and torch.equal(T["fit"]["mel"][0], A["tensors"]["target_mel"][0])
                 and torch.equal(T["fit"]["wave"][0], A["tensors"]["waveform"]) and torch.equal(T["positions"], A["tensors"]["positions"])
                 and torch.equal(T["mask"], A["tensors"]["mask"]))
    if not anchor_ok:
        raise SystemExit("STOP: TASK-24 rank-0 crop is not TASK-23's anchor pair bit for bit")
    b_sha = json.loads((REPO_ROOT / dc["task23_init_summary"]).read_text())["init_sha256"]
    seed_everything(cfg["init_seed"], torch.device("cpu"))
    fresh = MuQToMelTranslator(T["fit"]["features"].shape[-1], recipe["hidden_channels"], T["fit"]["mel"].shape[1])
    if not (init["sha256"] == b_sha == state_dict_sha256(fresh)):
        raise SystemExit("STOP: shared initial state is not TASK-23 B's step 0")
    n_in, n_mels, t_muq, t_mel = T["fit"]["features"].shape[-1], T["fit"]["mel"].shape[1], T["fit"]["features"].shape[1], T["fit"]["mel"].shape[-1]
    arch, base_out = {}, None
    for name, a in cfg["arms"].items():
        m = make_model(cfg, recipe, a, init, n_in, n_mels)
        with torch.no_grad():
            o = m(T["fit"]["features"][:1], t_mel, T["positions"])
        if not torch.isfinite(o).all() or o.shape != (1, n_mels, t_mel):
            raise SystemExit(f"STOP: arm {name} forward shape/finiteness failed")
        if name == "A":
            base_out = o
        macs = forward_macs(m, t_muq, t_mel)
        arch[name] = {"parameters": sum(p.numel() for p in m.parameters()), "init_sha256": state_dict_sha256(m),
                      "forward_macs_per_crop": macs, "receptive_field": source_receptive_field(m, T["positions"], t_muq),
                      "init_output_equals_A": bool(torch.equal(o, base_out))}
    if len({arch[k]["init_sha256"] for k in "ABCD"}) != 1 or not arch["F"]["init_output_equals_A"]:
        raise SystemExit("STOP: A-D initial weights differ, or F does not start at the current decoder's output")
    prep = {"ladder_pairs": {"path": dc["ladder_pairs"], "sha256": file_sha256(REPO_ROOT / dc["ladder_pairs"])},
            "init_state": {"path": dc["init_state"], "sha256": init["sha256"], "equals_task23_B": True},
            "anchor_bit_exact": True, "manifest": dc["manifest_csv"], "monitor_disjoint": True, "n_fit": len(fit_ids), "n_monitor": len(mon_ids),
            "architectures": arch, "geometry": {"t_muq": t_muq, "t_mel": t_mel, "n_in": n_in, "n_mels": n_mels}}
    atomic_write_text(out, json.dumps(prep, indent=2))
    print(f"prepared: A-D init {arch['A']['init_sha256'][:12]} (= TASK-23 B), F starts at A's output: {arch['F']['init_output_equals_A']}; "
          + ", ".join(f"{k}: {v['parameters']:,} params, {v['forward_macs_per_crop']['total'] / 1e6:.0f}M MACs" for k, v in arch.items()))
    return prep


# --- train -------------------------------------------------------------------------------

def spec_for(cfg: dict, recipe: dict, name: str, arm: dict, showcase: list[int]) -> ExposureSpec:
    b = cfg["budget"]
    return ExposureSpec(name=name, batch=arm["batch"], base_lr=float(arm["lr"]), betas=tuple(recipe["adam_betas"]),
                        total_presentations=b["total_presentations"], lr_drops=tuple((int(a), float(f)) for a, f in b["lr_drops"]),
                        log_every=b["log_every"], fit_eval_every=b["fit_eval_every"], monitor_eval_every=b["monitor_eval_every"],
                        latest_every=b["latest_every"], matched_updates=tuple(b["matched_updates"]), chunk=b["eval_chunk"],
                        sampler_seed=b["sampler_seed"], spike_factor=b["spike_factor"], recovery_factor=b["recovery_factor"],
                        anchor_index=0, showcase=tuple([0] + showcase))


def open_wandb(cfg: dict, run_name: str, arm_name: str, arm: dict, spec: ExposureSpec, env: dict, mode: str, resume_id: str | None):
    if mode == "disabled":
        return None
    from muq_inversion.wandb_logging import init_run

    kw = {"config": {"task": "TASK-25", "arm": arm_name, "arm_definition": arm, "spec": spec.__dict__, "run_name": run_name},
          "environment": env, "run_name": f"{run_name}_{arm_name}", "group": cfg["wandb_group"], "job_type": "ablation-arm",
          "tags": cfg["wandb_tags"] + [f"arm-{arm_name}", f"batch{arm['batch']}", f"lr{arm['lr']:g}", arm["arch"]], "resume_id": resume_id}
    try:
        return init_run(mode=mode, **kw)
    except Exception as exc:  # noqa: BLE001 -- logging must never stop training (CLAUDE.md rule 11)
        print(f"W&B {mode} init failed ({type(exc).__name__}: {exc}); continuing offline")
        return init_run(mode="offline", **kw)


def train_arms(cfg: dict, recipe: dict, device: torch.device, run_dir: Path, arms: list[str], stop_after: tuple[str, int] | None,
               steal: bool, stop: dict, env: dict, wandb_mode: str) -> int:
    ladder, init = load_inputs(cfg)
    T = ladder["tensors"]
    showcase = cfg["data"]["showcase_ranks"]
    t_mel, t_muq = T["fit"]["mel"].shape[-1], T["fit"]["features"].shape[1]
    positions = T["positions"].to(device)
    for name in arms:
        a = cfg["arms"][name]
        model = make_model(cfg, recipe, a, init, T["fit"]["features"].shape[-1], T["fit"]["mel"].shape[1]).to(device)
        boundary = boundary_frames(t_mel, (model.temporal_conv.receptive_field_frames - 1) // 2, T["positions"], t_muq)
        data = {"fit_feats": T["fit"]["features"].to(device), "fit_mel": T["fit"]["mel"].to(device), "mon_feats": T["monitor"]["features"].to(device),
                "mon_mel": T["monitor"]["mel"].to(device), "mask": T["mask"], "boundary": boundary, "positions": positions}
        spec = spec_for(cfg, recipe, name, a, showcase)
        arm_dir = run_dir / name
        lock = ArmLock(arm_dir / "LOCK", steal=steal)
        run = None
        try:
            if (arm_dir / "summary.json").exists():
                print(f"[{name}] already finished")
                continue
            latest = arm_dir / "checkpoints" / "latest.pt"
            resume_id = torch.load(latest, map_location="cpu", weights_only=False)["state"].get("wandb_run_id") if latest.exists() else None
            run = open_wandb(cfg, run_dir.name, name, a, spec, env, wandb_mode, resume_id)

            def sink(kind: str, row: dict, presentations: int, run=run) -> None:
                if run is None:
                    return
                try:
                    run.log({f"{kind}/{k}": v for k, v in row.items() if not isinstance(v, (list, dict))}
                            | ({f"{kind}/monitor_l1": row["monitor"]["l1"]} if "monitor" in row else {}), step=presentations)
                except Exception as exc:  # noqa: BLE001
                    print(f"W&B log failed: {type(exc).__name__}: {exc}")

            def hook(state: dict, run=run) -> None:
                if run is not None:
                    state["wandb_run_id"] = run.id

            print(f"=== arm {name}: {a} | {spec.updates:,} updates of batch {spec.batch} | LR drops at {spec.lr_drops} presentations")
            summ = train_exposure_arm(spec, model, data, arm_dir, device, meta={"arm": a, "init_sha256": state_dict_sha256(model)},
                                      stop_after_update=stop_after[1] if stop_after and stop_after[0] == name else None,
                                      should_stop=lambda: stop["flag"], metrics_sink=sink, state_hook=hook)
        finally:
            lock.release()
            if run is not None:
                run.finish()
        if summ is None:
            return 3
        print(f"[{name}] done: final fit L1 {summ['final']['l1']:.5f} (anchor {summ['final']['anchor_l1']:.5f}), monitor "
              f"{summ['final_monitor']['l1']:.4f}, {summ['runtime_seconds'] / 60:.1f} min, {summ['updates_per_second_median']:.0f} updates/s")
        if stop["flag"]:
            return 3
    return 0


# --- report ------------------------------------------------------------------------------

def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def report(cfg: dict, recipe: dict, device: torch.device, run_dir: Path, art: Path, env: dict, dev_info: dict) -> None:
    arms = list(cfg["arms"])
    prep = json.loads((run_dir / "prepared.json").read_text())
    S = {a: json.loads((run_dir / a / "summary.json").read_text()) for a in arms}
    fe = {a: jl(run_dir / a / "fit_eval.jsonl") for a in arms}
    me = {a: jl(run_dir / a / "monitor_eval.jsonl") for a in arms}
    tl = {a: jl(run_dir / a / "train_log.jsonl") for a in arms}
    b = cfg["budget"]
    drops = [x for x, _ in b["lr_drops"]]
    art.mkdir(parents=True, exist_ok=True)
    lbl = {a: f"{a}: {cfg['arms'][a]['arch']}{' x' + str(cfg['arms'][a].get('width_multiplier')) if cfg['arms'][a].get('width_multiplier') else ''}, "
              f"b{cfg['arms'][a]['batch']}, lr {cfg['arms'][a]['lr']:g}" for a in arms}

    # curves: fitting vs presentations / updates / wall time; monitoring separately
    for key, xl, fname, logx in (("presentations", "crop presentations (dotted: LR x0.1)", "fit_vs_presentations.png", False),
                                 ("update", "optimizer updates (log)", "fit_vs_updates.png", True),
                                 ("wall_seconds", "wall time (s)", "fit_vs_wall.png", False)):
        fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
        for a in arms:
            r = [x for x in fe[a] if x["update"] > 0]
            ax[0].plot([x[key] for x in r], [x["l1"] for x in r], color=COLORS[a], lw=1, label=lbl[a])
            m = [x for x in me[a] if x["update"] > 0]
            ax[1].plot([x[key] for x in m], [x["l1"] for x in m], color=COLORS[a], lw=1, marker=".", ms=3, label=lbl[a])
            if key == "presentations":
                for sp in S[a]["spikes"]:
                    ax[0].axvline(sp["presentations"], color=COLORS[a], ls="--", lw=0.5, alpha=0.6)
        for i, t in enumerate(("whole 256-crop fitting pool (dashed: spike events)", "MONITORING set (16 unseen-track crops; no selection)")):
            if key == "presentations":
                for d in drops:
                    ax[i].axvline(d, color="0.5", ls=":", lw=1)
            if logx:
                ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            ax[i].set_xlabel(xl)
            ax[i].set_ylabel("direct mel L1")
            ax[i].set_title(t)
            ax[i].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(art / fname, dpi=110)
        plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.5))
    for a in arms:
        ax[0].plot([x["presentations"] for x in tl[a]], [x["train_loss_window_max"] for x in tl[a]], color=COLORS[a], lw=0.5, label=a)
        ax[1].plot([x["presentations"] for x in tl[a]], [x["grad_norm_window_max"] for x in tl[a]], color=COLORS[a], lw=0.5, label=a)
    for i, t in enumerate(("train loss, max per 1,000-presentation window", "gradient norm, max per window")):
        for d in drops:
            ax[i].axvline(d, color="0.5", ls=":", lw=1)
        ax[i].set_yscale("log")
        ax[i].set_xlabel("crop presentations")
        ax[i].set_title(t)
        ax[i].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(art / "spikes_and_grad_norms.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for a in arms:
        r = [x for x in fe[a] if x["update"] > 0]
        ax.plot([x["presentations"] for x in r], [x["anchor_l1"] for x in r], color=COLORS[a], lw=1, label=lbl[a])
    for d in drops:
        ax.axvline(d, color="0.5", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("crop presentations")
    ax.set_ylabel("anchor-crop mel L1")
    ax.set_title("TASK-23 anchor crop")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(art / "anchor_vs_presentations.png", dpi=110)
    plt.close(fig)

    # final per-crop distributions and paired differences
    final_pc = {a: fe[a][-1]["per_crop_l1"] for a in arms}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot([final_pc[a] for a in arms], tick_labels=arms, whis=(0, 100))
    ax.set_yscale("log")
    ax.set_ylabel("per-crop mel L1 at 819,200 presentations (whiskers: min-max)")
    ax.set_title("Per-crop fitting error, final checkpoints")
    fig.tight_layout()
    fig.savefig(art / "per_crop_distribution.png", dpi=110)
    plt.close(fig)
    comps = []
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for axx, (x, y, what) in zip(axs.flat, cfg["comparisons"], strict=True):
        pdiff = paired_difference(final_pc[x], final_pc[y])
        pbest = paired_difference(S[x]["best"]["per_crop_l1"], S[y]["best"]["per_crop_l1"])
        pmon = paired_difference(me[x][-1]["per_crop_l1"], me[y][-1]["per_crop_l1"])
        comps.append({"a": x, "b": y, "isolates": what, "final_fit": pdiff, "best_fit": pbest, "final_monitor": pmon})
        lo, hi = min(final_pc[x] + final_pc[y]), max(final_pc[x] + final_pc[y])
        axx.scatter(final_pc[x], final_pc[y], s=6, alpha=0.6, color=COLORS[y])
        axx.plot([lo, hi], [lo, hi], "k:", lw=1)
        axx.set_xscale("log")
        axx.set_yscale("log")
        axx.set_xlabel(f"{x} per-crop L1")
        axx.set_ylabel(f"{y} per-crop L1")
        axx.set_title(f"{y} vs {x}: {what}\nmean diff {pdiff['mean']:+.4f} [{pdiff['ci95'][0]:+.4f}, {pdiff['ci95'][1]:+.4f}], "
                      f"{100 * pdiff['fraction_b_better']:.0f}% crops {y} better", fontsize=8)
    fig.tight_layout()
    fig.savefig(art / "paired_per_crop.png", dpi=110)
    plt.close(fig)

    # spectrograms + audio (anchor + pre-selected examples), best and final, every arm
    ladder, _ = load_inputs(cfg)
    T = ladder["tensors"]["fit"]
    rows = ladder["rows"]["fit"]
    bigvgan = load_bigvgan(device)
    h = bigvgan.h
    floor = bigvgan_log_mel_floor()
    showcase = [0] + cfg["data"]["showcase_ranks"]
    rc = cfg["report"]
    audio = {"reference": {}, "arms": {}}

    def vocode(mel: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return bigvgan(mel.clamp(min=floor)[None].to(device)).float().cpu()[0, 0]

    (art / "audio").mkdir(exist_ok=True)
    reg = {}
    for rank in showcase:
        cid = rows[rank]["crop_id"].replace(":", "_")
        reg[rank] = regions(T["mel"][rank][None], T["wave"][rank].numpy(), h.sampling_rate, h.hop_size, rc["zoom_frames"], rc["harmonic_max_bin"])
        audio["reference"][rank] = {"crop_id": rows[rank]["crop_id"],
                                    "original": write_wav(art / "audio" / f"r{rank}_{cid}_original.wav", T["wave"][rank], h.sampling_rate),
                                    "ground_truth_mel": write_wav(art / "audio" / f"r{rank}_{cid}_ground_truth_mel_bigvgan.wav", vocode(T["mel"][rank]), h.sampling_rate)}
    for a in arms:
        d = art / a
        d.mkdir(exist_ok=True)
        pr = run_dir / a / "preds"
        best, final = (torch.load(pr / f, map_location="cpu", weights_only=False) for f in ("best.pt", "final.pt"))
        audio["arms"][a] = {}
        for j, rank in enumerate(best["showcase"]):
            cid = rows[rank]["crop_id"].replace(":", "_")
            for label, pred, upd in (("best", best["fit"][j], best["update"]), ("final", final["fit"][rank], final["update"])):
                tgt = T["mel"][rank].numpy()
                title = f"TASK-25 {a}, rank {rank} ({rows[rank]['crop_id']}), {label} (update {upd:,})"
                spectrogram_figure(tgt, pred.numpy(), title, d / f"r{rank}_{label}_spectrogram.png")
                spectrogram_figure(tgt, pred.numpy(), f"{a}, rank {rank}, {label}: transient", d / f"r{rank}_{label}_zoom_transient.png",
                                   frames=tuple(reg[rank]["transient_window"]))
                spectrogram_figure(tgt, pred.numpy(), f"{a}, rank {rank}, {label}: harmonic region", d / f"r{rank}_{label}_zoom_harmonic.png",
                                   frames=tuple(reg[rank]["harmonic_window"]), bins=tuple(reg[rank]["harmonic_bins"]))
                if label == "final" and best["update"] == final["update"]:
                    continue
                audio["arms"][a][f"r{rank}_{label}"] = write_wav(d / f"r{rank}_{cid}_{label}_pred_mel_bigvgan.wav", vocode(pred), h.sampling_rate) | {
                    "update": upd, "direct_mel_l1": float((pred - T["mel"][rank]).abs().mean())}

    # tables
    def phase_median(a: str, lo: int, hi: int) -> float:
        v = [x["l1"] for x in fe[a] if lo < x["presentations"] <= hi]
        return float(np.median(v)) if v else float("nan")

    edges = [0] + drops + [b["total_presentations"]]
    win = b["total_presentations"] // 12  # last ~8% of the budget inside each phase
    on_aces = dev_info.get("resolved_device") == "xpu"
    arm_rows = {}
    for a in arms:
        s, ar = S[a], prep["architectures"][a]
        hrs = s["runtime_seconds"] / 3600
        arm_rows[a] = {"definition": cfg["arms"][a], "parameters": ar["parameters"], "forward_macs_per_crop": ar["forward_macs_per_crop"]["total"],
                       "train_macs_total_estimate": 3 * ar["forward_macs_per_crop"]["total"] * s["presentations"],
                       "receptive_field": ar["receptive_field"], "updates": s["updates_run"], "presentations": s["presentations"],
                       "runtime_hours": hrs, "updates_per_second_median": s["updates_per_second_median"],
                       "presentations_per_second_median": s["updates_per_second_median"] * cfg["arms"][a]["batch"],
                       "allocation_billing_hours": rc["billing_per_job_hour"] * hrs if on_aces else 0.0,
                       "final": s["final"], "final_monitor_l1": s["final_monitor"]["l1"], "best": {k: v for k, v in s["best"].items() if k != "per_crop_l1"},
                       "phase_medians": {f"phase{i + 1}_end": phase_median(a, max(edges[i], edges[i + 1] - win), edges[i + 1]) for i in range(len(edges) - 1)},
                       "lr_boundaries": s["lr_boundaries"], "matched_updates": s["matched_updates"],
                       "spikes": {"count": len(s["spikes"]), "max_peak_fit_l1": max((e.get("peak_fit_l1", e["fit_l1"]) for e in s["spikes"]), default=None),
                                  "recovery_presentations": [e.get("recovery_presentations") for e in s["spikes"]]},
                       "grad_norm_window_max_overall": s["grad_norm_window_max_overall"], "nonfinite_events": s["nonfinite_events"],
                       "presentation_counts_min_max": s["presentation_counts_min_max"]}
    rep = {"task": "TASK-25", "run_name": run_dir.name, "finished_utc": datetime.now(UTC).isoformat(), "device": dev_info, "environment": env,
           "config": cfg, "prepared": prep, "arms": arm_rows, "comparisons": comps, "audio": audio, "listening": "pending",
           "executed_on_aces": on_aces}
    atomic_write_text(art / "report.json", json.dumps(rep, indent=2, default=str))
    write_markdown(art, rep, cfg)
    print(f"report: {(art / 'TASK-25_RESULTS.md').relative_to(REPO_ROOT)}")


def write_markdown(art: Path, rep: dict, cfg: dict) -> None:
    R, b = rep["arms"], cfg["budget"]
    L = [f"# TASK-25 results: architecture vs optimization on the N=256 fitting pool (`{rep['run_name']}`)", "",
         (f"Executed on {rep['device']['device_name']}, commit {rep['environment'].get('git_commit')} (dirty={rep['environment'].get('git_dirty')}), "
         f"finished {rep['finished_utc']}. Direct pre-vocoder mel L1, evaluation mode, aggregated over every valid mel element. "
         f"Budget: {b['total_presentations']:,} presentations per arm ({b['total_presentations'] // 256:,} per crop); LR x0.1 at "
         f"{b['lr_drops'][0][0]:,} and {b['lr_drops'][1][0]:,} presentations. Single seed: exploratory."), "",
         "## Arms, size, compute, throughput", "",
         "| arm | arch | batch | LR | params | MACs/crop fwd | train MACs (≈3x fwd) | MuQ frames/output | updates | runtime | updates/s | presentations/s | billing-h |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a, r in R.items():
        d = r["definition"]
        L.append(f"| {a} | {d['arch']}{' x' + str(d['width_multiplier']) if d.get('width_multiplier') else ''} | {d['batch']} | {d['lr']:g} | {r['parameters']:,} | "
                 f"{r['forward_macs_per_crop'] / 1e6:.0f}M | {r['train_macs_total_estimate'] / 1e12:.0f}T | {r['receptive_field']['source_frames_per_output_min_median_max']} | "
                 f"{r['updates']:,} | {r['runtime_hours'] * 60:.1f} min | {r['updates_per_second_median']:.0f} | {r['presentations_per_second_median']:.0f} | {r['allocation_billing_hours']:.1f} |")
    L += ["", "## Primary comparison: final checkpoints at 819,200 presentations (fitting pool; monitoring separate)", "",
          "| arm | fit L1 | median | p90 | max | anchor | interior / boundary | dt / df (diag.) | monitoring | best fit L1 (presentations) | phase-end medians (1e-x LR phases) | spikes (max peak) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a, r in R.items():
        f, bst = r["final"], r["best"]
        ph = " / ".join(f"{v:.4f}" for v in r["phase_medians"].values())
        L.append(f"| {a} | {f['l1']:.4f} | {f['per_crop_median']:.4f} | {f['per_crop_p90']:.4f} | {f['per_crop_max']:.4f} | {f['anchor_l1']:.4f} | "
                 f"{f['l1_interior']:.4f} / {f['l1_boundary']:.4f} | {f['time_diff_l1']:.4f} / {f['freq_diff_l1']:.4f} | {r['final_monitor_l1']:.4f} | "
                 f"{bst['fit_l1']:.4f} ({bst['presentations']:,}) | {ph} | {r['spikes']['count']} ({r['spikes']['max_peak_fit_l1'] or 0:.2f}) |")
    L += ["", "## Paired per-crop differences (b - a over the 256 crops; negative = b fits better; bootstrap 95% CI over crops)", "",
          "| comparison | isolates | final fit: mean [CI], % crops b better | best fit: mean | final monitoring (16 crops): mean [CI] |", "| --- | --- | --- | --- | --- |"]
    for c in rep["comparisons"]:
        f, bb, m = c["final_fit"], c["best_fit"], c["final_monitor"]
        L.append(f"| {c['b']} vs {c['a']} | {c['isolates']} | {f['mean']:+.4f} [{f['ci95'][0]:+.4f}, {f['ci95'][1]:+.4f}], {100 * f['fraction_b_better']:.0f}% | "
                 f"{bb['mean']:+.4f} | {m['mean']:+.4f} [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}] |")
    L += ["", "## Secondary: matched optimizer updates (unequal exposure; possibly different LR phases)", "",
          "| arm | " + " | ".join(f"update {u:,}: presentations / LR / fit L1 / monitoring / wall" for u in b["matched_updates"]) + " |",
          "| --- | " + " | ".join("---" for _ in b["matched_updates"]) + " |"]
    for a, r in R.items():
        cells = []
        for u in b["matched_updates"]:
            mu = r["matched_updates"].get(str(u)) or r["matched_updates"].get(u)
            cells.append("-" if not mu or not mu["fit"] else f"{mu['presentations']:,} / {mu['lr']:g} / {mu['fit']['l1']:.4f} / "
                         f"{(mu['monitor'] or {}).get('l1', float('nan')):.4f} / {mu['fit']['wall_seconds'] / 60:.1f} min")
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L += ["", f"Historical context only (different schedule and budget): TASK-24 N=256 final fit L1 {cfg['report']['task24_n256_final_fit_l1']}.", "",
          "## Interpretation limits", "",
          "- Lower LR improving fit supports an optimization explanation under this budget. A low-LR plateau is not proof of convergence.",
          "- Batch-size effects at matched exposure come with 16x fewer updates (and different compute); they do not isolate gradient noise.",
          "- E over D supports a width/capacity contribution, but width also changes optimization and compute.",
          "- F over D supports the learned resampling parameterization; it does not prove interpolation was the sole bottleneck.",
          "- Lower training error without better monitoring error is a fitting improvement, not a demonstrated generalization improvement.",
          "- A null result does not establish representation information loss or exhausted architectural capacity. Single seed: exploratory.",
          "- Spikes are recorded, not attributed. Listening: **pending**.", "",
          ("Plots: `fit_vs_presentations.png`, `fit_vs_updates.png`, `fit_vs_wall.png` (fitting + monitoring panels), `spikes_and_grad_norms.png`, "
          "`anchor_vs_presentations.png`, `per_crop_distribution.png`, `paired_per_crop.png`; per arm: spectrograms, zooms, WAVs; references in `audio/`.")]
    atomic_write_text(art / "TASK-25_RESULTS.md", "\n".join(L) + "\n")


# --- main --------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "train", "report", "all"])
    parser.add_argument("--config", type=Path, default=Path("configs/task25.yaml"))
    parser.add_argument("--device", choices=DEVICE_CHOICES, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--arms", default=None, help="comma-separated arm names (default: all)")
    parser.add_argument("--stop-after", default=None, help="ARM:UPDATE -- simulate an interruption (resume test); exit 3")
    parser.add_argument("--steal-lock", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="torch.use_deterministic_algorithms(True) (CPU resume checks)")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()
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
    print(f"TASK-25 {args.stage} {args.run_name} | device {json.dumps(dev_info)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    mpath = run_dir / "run_manifest.json"
    if not mpath.exists():  # created once; concurrent arm processes each add their own start record below
        try:
            fd = os.open(mpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({"task": "TASK-25", "run_name": args.run_name, "config": cfg, "recipe": recipe,
                                    "uv_lock_sha256": file_sha256(REPO_ROOT / "uv.lock")}, indent=2, default=str))
        except FileExistsError:
            pass
    if json.loads(mpath.read_text())["config"] != json.loads(json.dumps(cfg, default=str)):
        raise SystemExit(f"STOP: {args.config} differs from the config this run started with")
    starts = run_dir / "starts"
    starts.mkdir(exist_ok=True)
    atomic_write_text(starts / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}_{os.getpid()}.json",
                      json.dumps({"stage": args.stage, "arms": args.arms, "device": dev_info, "environment": env, "argv": sys.argv,
                                  "deterministic_algorithms": args.deterministic, "wandb_mode": args.wandb_mode}, indent=2, default=str))
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    if args.stage in ("prepare", "all"):
        prepare(cfg, recipe, run_dir)
    elif not (run_dir / "prepared.json").exists():
        raise SystemExit("run the prepare stage first")
    if args.stage in ("train", "all"):
        arms = args.arms.split(",") if args.arms else list(cfg["arms"])
        bad = set(arms) - set(cfg["arms"])
        if bad:
            raise SystemExit(f"unknown arms {sorted(bad)}")
        sa = (args.stop_after.split(":")[0], int(args.stop_after.split(":")[1])) if args.stop_after else None
        code = train_arms(cfg, recipe, device, run_dir, arms, sa, args.steal_lock, stop, env, args.wandb_mode)
        if code:
            return code
    if args.stage in ("report", "all"):
        missing = [a for a in cfg["arms"] if not (run_dir / a / "summary.json").exists()]
        if missing:
            raise SystemExit(f"arms not finished: {missing}")
        report(cfg, recipe, device, run_dir, art, env, dev_info)
    return 0


if __name__ == "__main__":
    t0 = time.time()
    code = main()
    print(f"elapsed {time.time() - t0:.0f} s")
    raise SystemExit(code)
