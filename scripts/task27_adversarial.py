"""TASK-27: adversarial mel-spectrogram fine-tuning (L1-only vs L1 + hinge PatchGAN) -- single runner (tasks/TASK-27.md).

    uv run python scripts/task27_adversarial.py all       --config configs/task27.yaml --device xpu --run-name task27_v1
    uv run python scripts/task27_adversarial.py prepare   --config configs/task27.yaml --device xpu --run-name task27_v1
    uv run python scripts/task27_adversarial.py calibrate --config configs/task27.yaml --device xpu --run-name task27_v1
    uv run python scripts/task27_adversarial.py train     --config configs/task27.yaml --device xpu --run-name task27_v1 --arms A
    uv run python scripts/task27_adversarial.py report    --config configs/task27.yaml --device xpu --run-name task27_v1
    uv run python scripts/task27_adversarial.py score-listening --config configs/task27.yaml --run-name task27_v1

prepare: verifies TASK-24's cached pairs (per-crop hashes), manifest identity, monitoring disjointness and the TASK-23
  anchor; verifies the common start (TASK-25 C final: state-dict SHA-256 and its recorded final fit L1); describes the
  discriminator; fixes the discriminator input normalization constants and the 8 + 8 listening examples. No training.
calibrate: lambda from generator gradient norms on 8 fixed training batches after 100 D-only warm-up updates, with a
  temporary generator/discriminator. Stops if non-finite or degenerate. Writes calibration.json.
train: arms A (L1) and B (L1 + adversarial), each from freshly loaded start weights and freshly seeded state; resumable;
  one writer per arm (lock); one W&B run per arm.
report: plots, paired per-crop comparisons, conditioning sensitivity, spectrograms with shared scales and zooms, audio,
  the blinded listening package (separate key), report.json, TASK-27_RESULTS.md. Never overwrites a finished report.
score-listening: summarizes a filled-in ratings.csv with the key (never invents ratings).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
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
from muq_inversion.task24 import ArmLock, evaluate_set
from muq_inversion.task25 import paired_difference
from muq_inversion.task27 import (
    AdvSpec,
    build_discriminator,
    calibrate_lambda,
    d_input,
    logit_valid_mask,
    train_adversarial_arm,
    verify_start_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from task23_overfit import regions, resolve_recipe
from task24_ladder import load_ladder
from train_task18_heldout import load_config

COLORS = {"A": "C0", "B": "C3", "start": "0.4"}
CONDITIONS = ("start", "A_final", "B_final")


# --- shared setup --------------------------------------------------------------------------

def load_data(cfg: dict) -> dict:
    return load_ladder(REPO_ROOT / cfg["data"]["ladder_pairs"])  # per-crop hashes verified


def build_generator(recipe: dict, n_in: int, n_mels: int, state: dict) -> torch.nn.Module:
    g = MuQToMelTranslator(n_in, recipe["hidden_channels"], n_mels)
    g.load_state_dict(state)
    return g


def load_start(cfg: dict) -> dict:
    s = cfg["start"]
    path = REPO_ROOT / s["checkpoint"]
    if not path.exists():
        raise SystemExit(f"STOP: missing prerequisite: the common start checkpoint {s['checkpoint']} does not exist "
                         "(TASK-25 C's final checkpoint lives on ACES). No other model is substituted.")
    return torch.load(path, map_location="cpu", weights_only=False)


def prep_fn(prepared: dict):
    floor, c, s = prepared["discriminator_input"]["floor"], prepared["discriminator_input"]["center"], prepared["discriminator_input"]["scale"]
    return lambda mel: d_input(mel, floor, c, s)


def tensors_on(ladder: dict, device: torch.device, t_mel: int, t_muq: int, rf: int) -> dict:
    T = ladder["tensors"]
    return {"fit_feats": T["fit"]["features"].to(device), "fit_mel": T["fit"]["mel"].to(device),
            "mon_feats": T["monitor"]["features"].to(device), "mon_mel": T["monitor"]["mel"].to(device), "mask": T["mask"],
            "boundary": boundary_frames(t_mel, (rf - 1) // 2, T["positions"], t_muq), "positions": T["positions"].to(device)}


# --- prepare ----------------------------------------------------------------------------------

def prepare(cfg: dict, recipe: dict, device: torch.device, run_dir: Path) -> dict:
    out = run_dir / "prepared.json"
    if out.exists():
        print(f"prepared (cached): {out.relative_to(REPO_ROOT)}")
        return json.loads(out.read_text())
    dc = cfg["data"]
    ladder = load_data(cfg)
    T, meta = ladder["tensors"], ladder["meta"]
    with open(REPO_ROOT / dc["manifest_csv"], newline="") as f:
        man = list(csv.DictReader(f))
    fit_ids = [r["crop_id"] for r in man if r["role"] in ("anchor", "fit")]
    mon_ids = [r["crop_id"] for r in man if r["role"] == "monitor"]
    if [c["crop_id"] for c in meta["fit_crops"]] != fit_ids or [c["crop_id"] for c in meta["monitor_crops"]] != mon_ids:
        raise SystemExit("STOP: TASK-24 cached pairs do not match data/manifests/task24_ladder.csv")
    if len(fit_ids) != dc["n_fit"] or len(mon_ids) != dc["n_monitor"]:
        raise SystemExit("STOP: not TASK-25's 256 + 16 crops")
    if {r["track_id"] for r in man if r["role"] in ("anchor", "fit")} & {r["track_id"] for r in man if r["role"] == "monitor"}:
        raise SystemExit("STOP: monitoring tracks overlap the fitting pool")
    A = load_pair(REPO_ROOT / dc["anchor_pair"])
    if not (torch.equal(T["fit"]["features"][0], A["tensors"]["features"][0]) and torch.equal(T["fit"]["mel"][0], A["tensors"]["target_mel"][0])
            and torch.equal(T["positions"], A["tensors"]["positions"]) and torch.equal(T["mask"], A["tensors"]["mask"])):
        raise SystemExit("STOP: rank 0 is not TASK-23's anchor pair bit for bit")
    if not bool(T["mask"].all()):
        print("note: mask has invalid frames; discriminator logits touching them are masked")

    # common start
    s = cfg["start"]
    ck = load_start(cfg)
    n_in, n_mels, t_muq, t_mel = T["fit"]["features"].shape[-1], T["fit"]["mel"].shape[1], T["fit"]["features"].shape[1], T["fit"]["mel"].shape[-1]
    g = build_generator(recipe, n_in, n_mels, ck["model_state_dict"])
    try:
        sha = verify_start_checkpoint(g, ck, s)
    except ValueError as exc:
        raise SystemExit(f"STOP: {exc}") from exc
    g = g.to(device)
    data = tensors_on(ladder, device, t_mel, t_muq, g.temporal_conv.receptive_field_frames)
    fm, _, _ = evaluate_set(g, data["fit_feats"], data["fit_mel"], data["mask"], data["boundary"], data["positions"], device)
    if abs(fm["l1"] - s["expected_fit_l1"]) > s["fit_l1_atol"]:
        raise SystemExit(f"STOP: start checkpoint fit L1 {fm['l1']:.7f} != recorded {s['expected_fit_l1']:.7f} (atol {s['fit_l1_atol']})")
    g2 = build_generator(recipe, n_in, n_mels, ck["model_state_dict"])
    if state_dict_sha256(g2) != sha:
        raise SystemExit("STOP: the two arms would not start from identical generator weights")

    # discriminator description and input normalization (identical for real and generated mels)
    floor = bigvgan_log_mel_floor()
    tg = T["fit"]["mel"].clamp(min=floor)
    din = {"floor": floor, "center": float(tg.mean()), "scale": float(tg.std()), "source": "mean/std of the 256 clamped training targets"}
    hop_ms = 1000 * cfg["report"]["hop_size"] / cfg["report"]["sample_rate"]
    d = build_discriminator(cfg["discriminator"], n_in, cfg["discriminator"]["init_seed"])
    ddesc = d.describe(n_mels, t_mel, hop_ms)
    if not ddesc["fits_crop"]:
        raise SystemExit(f"STOP: discriminator receptive field {ddesc['receptive_field_frames']} does not fit the crop")
    lv = logit_valid_mask(T["mask"], n_mels)
    with torch.no_grad():
        o = d(d_input(T["fit"]["mel"][:2], floor, din["center"], din["scale"]), torch.zeros(2, n_in, t_mel))
    if tuple(o.shape[1:]) != tuple(lv.shape[1:]):
        raise SystemExit(f"STOP: logit grid {tuple(o.shape)} != valid-mask grid {tuple(lv.shape)}")

    # listening examples, fixed before any training
    rng = random.Random(cfg["listening"]["selection_seed"])
    fixed = cfg["listening"]["training_fixed_ranks"]
    rest = [i for i in range(len(fit_ids)) if i not in fixed]
    train_sel = fixed + sorted(rng.sample(rest, cfg["listening"]["n_training"] - len(fixed)))
    mon_sel = sorted(rng.sample(range(len(mon_ids)), cfg["listening"]["n_monitoring"]))
    prep = {"ladder_pairs": {"path": dc["ladder_pairs"], "sha256": file_sha256(REPO_ROOT / dc["ladder_pairs"])},
            "manifest": dc["manifest_csv"], "anchor_bit_exact": True, "monitor_disjoint": True, "n_fit": len(fit_ids), "n_monitor": len(mon_ids),
            "mask_all_valid": bool(T["mask"].all()),
            "start": {"checkpoint": s["checkpoint"], "file_sha256": file_sha256(REPO_ROOT / s["checkpoint"]), "state_dict_sha256": sha,
                      "fit_l1_measured": fm["l1"], "fit_l1_recorded": s["expected_fit_l1"], "standin": bool(s.get("standin", False)),
                      "note": s.get("note", "")},
            "generator": {"parameters": sum(p.numel() for p in g.parameters()), "identical_start_for_both_arms": True},
            "discriminator": ddesc, "discriminator_input": din, "logit_valid_fraction": float(lv.float().mean()),
            "geometry": {"t_muq": t_muq, "t_mel": t_mel, "n_in": n_in, "n_mels": n_mels},
            "listening": {"training_ranks": train_sel, "training_crop_ids": [fit_ids[i] for i in train_sel],
                          "monitoring_indices": mon_sel, "monitoring_crop_ids": [mon_ids[i] for i in mon_sel]}}
    atomic_write_text(out, json.dumps(prep, indent=2))
    print(f"prepared: start {sha[:12]} (fit L1 {fm['l1']:.6f}{', STAND-IN' if prep['start']['standin'] else ''}); "
          f"D {ddesc['parameters']:,} params, RF {ddesc['receptive_field_frames']} frames = {ddesc['receptive_field_ms']:.0f} ms x "
          f"{ddesc['receptive_field_bins']} bins, logits {ddesc['logit_grid']}; listening train {train_sel}, monitor {mon_sel}")
    return prep


# --- calibrate ----------------------------------------------------------------------------------

def calibrate(cfg: dict, recipe: dict, device: torch.device, run_dir: Path, prepared: dict) -> dict:
    out = run_dir / "calibration.json"
    if out.exists():
        print(f"calibration (cached): {out.relative_to(REPO_ROOT)}")
        return json.loads(out.read_text())
    ladder = load_data(cfg)
    T = ladder["tensors"]
    geo = prepared["geometry"]
    g = build_generator(recipe, geo["n_in"], geo["n_mels"], load_start(cfg)["model_state_dict"]).to(device)
    d = build_discriminator(cfg["discriminator"], geo["n_in"], cfg["discriminator"]["init_seed"]).to(device)
    tr = cfg["training"]
    opt_d = torch.optim.Adam(d.parameters(), lr=tr["d_lr"], betas=tuple(tr["d_betas"]), weight_decay=0.0)
    data = tensors_on(ladder, device, geo["t_mel"], geo["t_muq"], g.temporal_conv.receptive_field_frames)
    valid = logit_valid_mask(T["mask"], geo["n_mels"]).to(device)
    t0 = time.perf_counter()
    res = calibrate_lambda(g, d, opt_d, data, cfg["calibration"], tr["d_lr"], tr["batch"], prep_fn(prepared), valid, device)
    res["seconds"] = time.perf_counter() - t0
    res["start_sha256"] = state_dict_sha256(g)
    atomic_write_text(out, json.dumps(res, indent=2))
    if res["lambda"] is None:
        raise SystemExit(f"STOP: lambda calibration is degenerate: {res['problems']} (see {out.relative_to(REPO_ROOT)})")
    print(f"calibration: median |g_L1| {res['median_grad_norm_l1']:.4g}, median |g_adv| {res['median_grad_norm_adv']:.4g} -> "
          f"lambda {res['lambda']:.4g} (weighted adv / L1 median {res['weighted_adv_over_l1_median']:.3f}); D warm-up loss "
          f"{res['warmup']['first']['d_loss']:.3f} -> {res['warmup']['last']['d_loss']:.3f}")
    return res


# --- train ----------------------------------------------------------------------------------------

def spec_for(cfg: dict, name: str, lam: float) -> AdvSpec:
    tr, a = cfg["training"], cfg["arms"][name]
    return AdvSpec(name=name, adversarial=a["adversarial"], batch=tr["batch"], base_lr=float(tr["g_lr"]), betas=tuple(tr["g_betas"]),
                   total_presentations=tr["total_presentations"], lr_drops=tuple((int(x), float(f)) for x, f in tr["lr_drops"]),
                   lam=lam if a["adversarial"] else 0.0, ramp_fraction=tr["lambda_ramp_fraction"], d_lr=float(tr["d_lr"]),
                   d_betas=tuple(tr["d_betas"]), d_warmup_updates=cfg["calibration"]["warmup_updates"],
                   d_warmup_sampler_seed=cfg["calibration"]["warmup_sampler_seed"], d_init_seed=cfg["discriminator"]["init_seed"],
                   log_every=tr["log_every"], fit_eval_every=tr["fit_eval_every"], monitor_eval_every=tr["monitor_eval_every"],
                   latest_every=tr["latest_every"], chunk=tr["eval_chunk"], sampler_seed=tr["sampler_seed"],
                   spike_factor=tr["spike_factor"], recovery_factor=tr["recovery_factor"], showcase=tuple(cfg["report"]["showcase_ranks"]))


def open_wandb(cfg: dict, run_name: str, arm: str, spec: AdvSpec, env: dict, mode: str, resume_id: str | None, extra: dict):
    if mode == "disabled":
        return None
    from muq_inversion.wandb_logging import init_run

    kw = {"config": {"task": "TASK-27", "arm": arm, "arm_definition": cfg["arms"][arm], "spec": spec.__dict__, "run_name": run_name} | extra,
          "environment": env, "run_name": f"{run_name}_{arm}", "group": cfg["wandb_group"], "job_type": "adversarial-arm",
          "tags": cfg["wandb_tags"] + [f"arm-{arm}", "adversarial" if spec.adversarial else "l1-only"], "resume_id": resume_id}
    try:
        return init_run(mode=mode, **kw)
    except Exception as exc:  # noqa: BLE001 -- logging must never stop training (CLAUDE.md rule 11)
        print(f"W&B {mode} init failed ({type(exc).__name__}: {exc}); continuing offline")
        return init_run(mode="offline", **kw)


def train_arms(cfg: dict, recipe: dict, device: torch.device, run_dir: Path, prepared: dict, arms: list[str],
               stop_after: tuple[str, int] | None, steal: bool, stop: dict, env: dict, wandb_mode: str) -> int:
    cal = json.loads((run_dir / "calibration.json").read_text()) if (run_dir / "calibration.json").exists() else None
    if any(cfg["arms"][a]["adversarial"] for a in arms) and (cal is None or cal.get("lambda") is None):
        raise SystemExit("STOP: run the calibrate stage first (arm B needs the calibrated lambda)")
    ladder = load_data(cfg)
    T = ladder["tensors"]
    geo = prepared["geometry"]
    start_state = load_start(cfg)["model_state_dict"]
    valid = logit_valid_mask(T["mask"], geo["n_mels"])
    for name in arms:
        seed_everything(cfg["seed"], device)  # identical fresh state for every arm (after calibration)
        g = build_generator(recipe, geo["n_in"], geo["n_mels"], start_state).to(device)
        if state_dict_sha256(g) != prepared["start"]["state_dict_sha256"]:
            raise SystemExit(f"STOP: arm {name} would not start from the verified weights")
        spec = spec_for(cfg, name, cal["lambda"] if cal and cal.get("lambda") is not None else 0.0)
        d = build_discriminator(cfg["discriminator"], geo["n_in"], cfg["discriminator"]["init_seed"]).to(device) if spec.adversarial else None
        data = tensors_on(ladder, device, geo["t_mel"], geo["t_muq"], g.temporal_conv.receptive_field_frames)
        arm_dir = run_dir / name
        lock = ArmLock(arm_dir / "LOCK", steal=steal)
        run = None
        try:
            if (arm_dir / "summary.json").exists():
                print(f"[{name}] already finished")
                continue
            latest = arm_dir / "checkpoints" / "latest.pt"
            resume_id = torch.load(latest, map_location="cpu", weights_only=False)["state"].get("wandb_run_id") if latest.exists() else None
            run = open_wandb(cfg, run_dir.name, name, spec, env, wandb_mode, resume_id,
                             {"calibration": {k: v for k, v in (cal or {}).items() if k != "batches"}, "start": prepared["start"],
                              "discriminator": prepared["discriminator"]})

            def sink(kind: str, row: dict, presentations: int, run=run) -> None:
                if run is None:
                    return
                try:
                    flat = {f"{kind}/{k}": v for k, v in row.items() if isinstance(v, (int, float)) and v is not None}
                    if "monitor" in row:
                        flat |= {f"{kind}/monitor_{k}": v for k, v in row["monitor"].items() if isinstance(v, (int, float))}
                    run.log(flat, step=presentations)
                except Exception as exc:  # noqa: BLE001
                    print(f"W&B log failed: {type(exc).__name__}: {exc}")

            def hook(state: dict, run=run) -> None:
                if run is not None:
                    state["wandb_run_id"] = run.id

            print(f"=== arm {name}: {cfg['arms'][name]} | {spec.updates:,} generator updates of batch {spec.batch} | LR drops at "
                  f"{spec.lr_drops} presentations | lambda {spec.lam:.4g} (ramp {spec.ramp_updates:,} updates)")
            summ = train_adversarial_arm(spec, g, d, data, arm_dir, device, prep_fn(prepared), valid,
                                         meta={"arm": cfg["arms"][name], "start": prepared["start"], "lambda_calibrated": spec.lam},
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


# --- report ------------------------------------------------------------------------------------

def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def panel_figure(mels: dict[str, np.ndarray], title: str, path: Path, frames: tuple[int, int] | None = None,
                 bins: tuple[int, int] | None = None) -> None:
    """Every panel on one shared color scale (set by the ground-truth target's range in the window)."""
    tgt = next(iter(mels.values()))
    f0, f1 = frames or (0, tgt.shape[1])
    b0, b1 = bins or (0, tgt.shape[0])
    vmin, vmax = float(tgt[b0:b1, f0:f1].min()), float(tgt[b0:b1, f0:f1].max())
    fig, ax = plt.subplots(1, len(mels), figsize=(4 * len(mels), 3.6), sharex=True, sharey=True)
    ext = (f0 - 0.5, f1 - 0.5, b0 - 0.5, b1 - 0.5)
    for a, (name, m) in zip(ax, mels.items(), strict=True):
        im = a.imshow(m[b0:b1, f0:f1], origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="magma", extent=ext)
        a.set_title(name, fontsize=9)
        a.set_xlabel("mel frame (93.75 /s)")
    ax[0].set_ylabel("mel bin")
    fig.colorbar(im, ax=list(ax), label="log-mel (nats), shared scale (target range)", fraction=0.02)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def report(cfg: dict, recipe: dict, device: torch.device, run_dir: Path, art: Path, env: dict, dev_info: dict) -> None:
    prepared = json.loads((run_dir / "prepared.json").read_text())
    cal = json.loads((run_dir / "calibration.json").read_text())
    arms = list(cfg["arms"])
    S = {a: json.loads((run_dir / a / "summary.json").read_text()) for a in arms}
    if len({S[a]["start_sha256"] for a in arms}) != 1 or len({S[a]["order_sha256"] for a in arms}) != 1:
        raise SystemExit("STOP: arms did not share the start weights and the exact sample order")
    fe = {a: jl(run_dir / a / "fit_eval.jsonl") for a in arms}
    me = {a: jl(run_dir / a / "monitor_eval.jsonl") for a in arms}
    tl = {a: jl(run_dir / a / "train_log.jsonl") for a in arms}
    tr = cfg["training"]
    drops = [x for x, _ in tr["lr_drops"]]
    art.mkdir(parents=True, exist_ok=True)

    # curves
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a in arms:
        ax[0].plot([x["presentations"] for x in fe[a]], [x["l1"] for x in fe[a]], color=COLORS[a], label=f"{a} fit")
        ax[1].plot([x["presentations"] for x in me[a]], [x["l1"] for x in me[a]], color=COLORS[a], marker=".", label=f"{a} monitoring")
        ax[2].plot([x["presentations"] for x in me[a]], [x["shuffled_conditioning_l1"] - x["l1"] for x in me[a]], color=COLORS[a], marker=".",
                   label=f"{a} monitoring")
        ax[2].plot([x["presentations"] for x in fe[a] if "shuffled_conditioning_l1" in x],
                   [x["shuffled_conditioning_l1"] - x["l1"] for x in fe[a] if "shuffled_conditioning_l1" in x], color=COLORS[a], ls="--", label=f"{a} fit")
    for i, t in enumerate(("256-crop fitting pool (training)", "16 monitoring crops (unseen tracks; no selection)",
                           "conditioning sensitivity: shuffled - correct L1")):
        for x in drops:
            ax[i].axvline(x, color="0.6", ls=":")
        ax[i].axvline(S["B"]["spec"]["ramp_fraction"] * tr["total_presentations"], color="C3", ls=":", lw=0.8)
        ax[i].set_title(t)
        ax[i].set_xlabel("generator crop presentations")
        ax[i].legend(fontsize=8)
    ax[0].set_ylabel("direct masked mel L1")
    fig.tight_layout()
    fig.savefig(art / "curves.png", dpi=110)
    plt.close(fig)
    tb = tl["B"]
    fig, ax = plt.subplots(2, 3, figsize=(18, 8))
    x = [r["presentations"] for r in tb]
    ax[0, 0].plot(x, [r["d_loss_real_window_mean"] for r in tb], label="D real hinge")
    ax[0, 0].plot(x, [r["d_loss_fake_window_mean"] for r in tb], label="D fake hinge")
    ax[0, 0].set_title("discriminator losses (window means)")
    ax[0, 1].plot(x, [r["d_real_logit_mean_window_mean"] for r in tb], label="real")
    ax[0, 1].plot(x, [r["d_fake_logit_mean_window_mean"] for r in tb], label="generated")
    ax[0, 1].set_title("mean logits (training batches)")
    ax[0, 2].plot(x, [r["d_real_saturated_frac_window_mean"] for r in tb], label="real logits >= 1")
    ax[0, 2].plot(x, [r["d_fake_saturated_frac_window_mean"] for r in tb], label="generated logits <= -1")
    ax[0, 2].plot(x, [r["d_accuracy_window_mean"] for r in tb], label="sign accuracy")
    ax[0, 2].set_title("saturation / accuracy")
    ax[1, 0].plot(x, [r["lambda"] for r in tb])
    ax[1, 0].set_title("lambda (ramp then fixed)")
    ax[1, 0].plot([], [], label="lambda")
    for a in arms:
        ax[1, 1].plot([r["presentations"] for r in tl[a]], [r["g_grad_norm_l1"] for r in tl[a]], color=COLORS[a], lw=0.6, label=f"{a} |g_L1|")
    ax[1, 1].plot(x, [r["lambda"] * r["g_grad_norm_adv"] for r in tb], color="k", lw=0.6, label="B lambda |g_adv|")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("generator gradient norms (at log steps)")
    ax[1, 2].plot(x, [r["d_grad_norm_window_max"] for r in tb], label="D grad norm (window max)")
    ax[1, 2].set_yscale("log")
    ax[1, 2].set_title("discriminator gradient norm")
    for a_ in ax.flat:
        a_.set_xlabel("generator crop presentations")
        a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(art / "adversarial_dynamics.png", dpi=110)
    plt.close(fig)

    # paired per-crop comparisons (B - A; negative = B lower error)
    final_fit = {a: fe[a][-1]["per_crop_l1"] for a in arms}
    final_mon = {a: me[a][-1]["per_crop_l1"] for a in arms}
    comps = {"final_fit": paired_difference(final_fit["A"], final_fit["B"]), "final_monitor": paired_difference(final_mon["A"], final_mon["B"]),
             "best_fit": paired_difference(S["A"]["best"]["per_crop_l1"], S["B"]["best"]["per_crop_l1"]),
             "start_to_A_fit": paired_difference(fe["A"][0]["per_crop_l1"], final_fit["A"]),
             "start_to_B_fit": paired_difference(fe["B"][0]["per_crop_l1"], final_fit["B"]),
             "start_to_A_monitor": paired_difference(me["A"][0]["per_crop_l1"], final_mon["A"]),
             "start_to_B_monitor": paired_difference(me["B"][0]["per_crop_l1"], final_mon["B"])}
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i, (k, t) in enumerate((("final_fit", "fitting pool (256)"), ("final_monitor", "monitoring (16)"))):
        va, vb = (final_fit if i == 0 else final_mon)["A"], (final_fit if i == 0 else final_mon)["B"]
        lo, hi = min(va + vb), max(va + vb)
        ax[i].scatter(va, vb, s=8, alpha=0.6, color="C3")
        ax[i].plot([lo, hi], [lo, hi], "k:")
        c = comps[k]
        ax[i].set_title(f"{t}: B - A {c['mean']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}], {100 * c['fraction_b_better']:.0f}% B lower", fontsize=9)
        ax[i].set_xlabel("A (L1 only) per-crop L1")
        ax[i].set_ylabel("B (L1 + adversarial) per-crop L1")
    fig.tight_layout()
    fig.savefig(art / "paired_per_crop.png", dpi=110)
    plt.close(fig)

    # predictions, spectrograms, audio, blinded listening package
    ladder = load_data(cfg)
    T = ladder["tensors"]
    rows = ladder["rows"]
    preds = {"start": torch.load(run_dir / "A" / "preds" / "start.pt", weights_only=False)}
    start_b = torch.load(run_dir / "B" / "preds" / "start.pt", weights_only=False)
    if not torch.equal(preds["start"]["fit"], start_b["fit"]):
        print("note: start predictions of A and B differ by accelerator nondeterminism; A's are used for the 'start' condition")
    preds["A_final"] = torch.load(run_dir / "A" / "preds" / "final.pt", weights_only=False)
    preds["B_final"] = torch.load(run_dir / "B" / "preds" / "final.pt", weights_only=False)
    bigvgan = load_bigvgan(device)
    h = bigvgan.h
    floor = bigvgan_log_mel_floor()

    def vocode(mel: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            return bigvgan(mel.float().clamp(min=floor)[None].to(device)).float().cpu()[0, 0].numpy()

    L = prepared["listening"]
    examples = [("training", r, rows["fit"][r]["crop_id"], "fit") for r in L["training_ranks"]] + \
               [("monitoring", i, rows["monitor"][i]["crop_id"], "monitor") for i in L["monitoring_indices"]]
    spec_dir = art / "spectrograms"
    spec_dir.mkdir(exist_ok=True)
    rc = cfg["report"]
    clips, meta_ex = [], []
    for k, (group, i, cid, key) in enumerate(examples, start=1):
        tgt = T[key]["mel"][i]
        wave = T[key]["wave"][i].numpy()
        mels = {"ground truth": tgt.numpy()} | {c: preds[c][key][i].numpy() for c in CONDITIONS}
        reg = regions(tgt[None], wave, h.sampling_rate, h.hop_size, rc["zoom_frames"], rc["harmonic_max_bin"])
        stem = f"ex{k:02d}_{group}_{cid.replace(':', '_')}"
        l1s = {c: float((preds[c][key][i] - tgt).abs().mean()) for c in CONDITIONS}
        ttl = f"{group} {cid}: L1 start {l1s['start']:.3f}, A {l1s['A_final']:.3f}, B {l1s['B_final']:.3f}"
        panel_figure(mels, ttl, spec_dir / f"{stem}_full.png")
        panel_figure(mels, ttl + " (transient zoom)", spec_dir / f"{stem}_zoom_transient.png", frames=tuple(reg["transient_window"]))
        panel_figure(mels, ttl + " (harmonic zoom)", spec_dir / f"{stem}_zoom_harmonic.png", frames=tuple(reg["harmonic_window"]),
                     bins=tuple(reg["harmonic_bins"]))
        audio = {"original": wave, "ground_truth_mel": vocode(tgt)} | {c: vocode(preds[c][key][i]) for c in CONDITIONS}
        clips.append(audio)
        meta_ex.append({"example": k, "group": group, "index": i, "crop_id": cid, "direct_mel_l1": l1s, "regions": reg})
    peak = max(float(np.abs(a).max()) for ex in clips for a in ex.values())
    gain = min(1.0, rc["listening_peak"] / peak)
    blind, key_dir = art / "listening_blind", art / "listening_blind_KEY"
    blind.mkdir(exist_ok=True)
    key_dir.mkdir(exist_ok=True)
    rng = random.Random(cfg["listening"]["blind_seed"])
    key, sheet = {"common_gain": gain, "max_peak_before_gain": peak, "examples": {}}, []
    for ex, m in zip(clips, meta_ex, strict=True):
        d = blind / f"example_{m['example']:02d}"
        d.mkdir(exist_ok=True)
        sf.write(d / "reference_original.wav", ex["original"] * gain, h.sampling_rate, subtype="FLOAT")
        sf.write(d / "reference_ground_truth_mel_bigvgan.wav", ex["ground_truth_mel"] * gain, h.sampling_rate, subtype="FLOAT")
        order = list(CONDITIONS)
        rng.shuffle(order)
        key["examples"][f"example_{m['example']:02d}"] = {"clip_1": order[0], "clip_2": order[1], "clip_3": order[2]} | {
            "group": m["group"], "crop_id": m["crop_id"]}
        for j, c in enumerate(order, start=1):
            sf.write(d / f"clip_{j}.wav", ex[c] * gain, h.sampling_rate, subtype="FLOAT")
            sheet.append({"example": f"example_{m['example']:02d}", "clip": f"clip_{j}", "quality_artifacts_1to5": "",
                          "transient_sharpness_1to5": "", "pitch_timbre_fidelity_1to5": "", "notes": ""})
    with open(blind / "ratings.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(sheet[0]))
        wr.writeheader()
        wr.writerows(sheet)
    with open(blind / "preferences.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["example", "preferred_clip_overall", "notes"])
        wr.writeheader()
        wr.writerows({"example": e, "preferred_clip_overall": "", "notes": ""} for e in key["examples"])
    atomic_write_text(blind / "README.txt", "\n".join([
        "TASK-27 blinded listening package (do NOT open ../listening_blind_KEY until all ratings are entered).",
        f"{len(clips)} examples: 8 from the 256 training crops and 8 from the 16 unseen-track monitoring crops (mixed order is not",
        "used; examples 1-8 are training, 9-16 monitoring). Each folder has two labeled references and three blinded clips:",
        "  reference_original.wav                 the original 2-s crop",
        "  reference_ground_truth_mel_bigvgan.wav  BigVGAN applied to the true mel (vocoder reference)",
        "  clip_1.wav, clip_2.wav, clip_3.wav     three reconstructions in random order",
        f"All files share one playback gain ({gain:.4f}); nothing is loudness-normalized per file.",
        "",
        "ratings.csv: for each clip, 1 (worst) to 5 (best):",
        "  quality_artifacts_1to5        overall quality / absence of artifacts",
        "  transient_sharpness_1to5      sharpness of attacks and transients",
        "  pitch_timbre_fidelity_1to5    pitch and timbre fidelity to reference_original",
        "preferences.csv: preferred_clip_overall = clip_1, clip_2, clip_3 (or 'none').",
        "Then: uv run python scripts/task27_adversarial.py score-listening --config <config> --run-name <run>"]) + "\n")
    atomic_write_text(key_dir / "key.json", json.dumps(key, indent=2))

    # tables
    on_aces = dev_info.get("resolved_device") == "xpu"
    arm_rows = {}
    for a in arms:
        s = S[a]
        hrs = s["runtime_seconds"] / 3600
        arm_rows[a] = {"definition": cfg["arms"][a], "lambda": s["spec"]["lam"], "updates": s["updates_run"], "presentations": s["presentations"],
                       "runtime_hours": hrs, "updates_per_second_median": s["updates_per_second_median"],
                       "allocation_billing_hours": rc["billing_per_job_hour"] * hrs if on_aces else 0.0,
                       "start": s["start"], "start_monitor": s["start_monitor"], "final": s["final"], "final_monitor": s["final_monitor"],
                       "best": {k: v for k, v in s["best"].items() if k != "per_crop_l1"}, "spikes": s["spikes"],
                       "nonfinite_events": s["nonfinite_events"], "presentation_counts_min_max": s["presentation_counts_min_max"],
                       "g_grad_norm_window_max_overall": s["g_grad_norm_window_max_overall"],
                       "d_warmup": s["d_warmup"], "final_train_log": tl[a][-1] if tl[a] else None}
    g_macs = 201_818_624
    d_macs = prepared["discriminator"]["forward_macs_per_crop"]
    compute = {"generator_forward_macs_per_crop": g_macs, "discriminator_forward_macs_per_crop": d_macs,
               "A_train_macs_estimate": 3 * g_macs * tr["total_presentations"],
               "B_train_macs_estimate": 3 * g_macs * tr["total_presentations"]
               + 3 * d_macs * 2 * tr["total_presentations"]        # D step: real + generated
               + 3 * d_macs * tr["total_presentations"]            # generator step through D
               + 3 * g_macs * tr["total_presentations"] * 0.5      # second generator backward (separate gradient terms)
               + 3 * d_macs * 2 * cfg["calibration"]["warmup_updates"] * tr["batch"],
               "note": "analytic estimates (backward ~2x forward); B also runs 100 D-only warm-up updates; wall times are measured"}
    rep = {"task": "TASK-27", "run_name": run_dir.name, "finished_utc": datetime.now(UTC).isoformat(), "device": dev_info, "environment": env,
           "config": cfg, "prepared": prepared, "calibration": cal, "arms": arm_rows, "comparisons": comps, "compute": compute,
           "listening": {"package": str(blind.relative_to(REPO_ROOT)), "key": str(key_dir.relative_to(REPO_ROOT)), "ratings": "pending",
                         "examples": meta_ex, "common_gain": gain}, "executed_on_aces": on_aces}
    atomic_write_text(art / "report.json", json.dumps(rep, indent=2, default=str))
    write_markdown(art, rep)
    print(f"report: {(art / 'TASK-27_RESULTS.md').relative_to(REPO_ROOT)}")


def write_markdown(art: Path, rep: dict) -> None:
    R, P, C, cal = rep["arms"], rep["prepared"], rep["comparisons"], rep["calibration"]
    st = P["start"]
    L = [f"# TASK-27 results: adversarial mel reconstruction (`{rep['run_name']}`)", "",
         (f"Executed on {rep['device']['device_name']}, commit {rep['environment'].get('git_commit')} (dirty={rep['environment'].get('git_dirty')}), "
          f"finished {rep['finished_utc']}. Direct masked mel L1 (evaluation mode). Single seed, reused monitoring set: exploratory."), "",
         (f"**Common start:** `{st['checkpoint']}` (state-dict SHA-256 `{st['state_dict_sha256'][:16]}...`, fit L1 {st['fit_l1_measured']:.6f}; "
          f"recorded {st['fit_l1_recorded']:.6f})" + (" -- **STAND-IN, not TASK-25 C's ACES checkpoint**" if st["standin"] else "") + "."),
         (f"**Discriminator:** {P['discriminator']['parameters']:,} parameters; receptive field {P['discriminator']['receptive_field_bins']} bins x "
          f"{P['discriminator']['receptive_field_frames']} frames ({P['discriminator']['receptive_field_ms']:.0f} ms); logit grid {P['discriminator']['logit_grid']}."),
         (f"**Lambda:** {cal['lambda']:.4g} (median |g_L1| {cal['median_grad_norm_l1']:.4g}, median |g_adv| {cal['median_grad_norm_adv']:.4g}, "
          f"target ratio {cal['target_ratio']}), ramped over the first 10% of generator updates."), "",
         "## Final checkpoints at matched generator exposure (primary)", "",
         "| arm | fit L1 | median / p90 / max | anchor | interior / boundary | dt / df diff | shuffled-cond. fit | monitoring | shuffled-cond. monitoring | runtime | updates/s |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a, r in [("start", None)] + list(R.items()):
        if r is None:
            f, m = R["A"]["start"], R["A"]["start_monitor"]
            L.append(f"| start (TASK-25 C) | {f['l1']:.4f} | {f['per_crop_median']:.4f} / {f['per_crop_p90']:.4f} / {f['per_crop_max']:.4f} | "
                     f"{f['anchor_l1']:.4f} | {f['l1_interior']:.4f} / {f['l1_boundary']:.4f} | {f['time_diff_l1']:.4f} / {f['freq_diff_l1']:.4f} | "
                     f"{f.get('shuffled_conditioning_l1', float('nan')):.4f} | {m['l1']:.4f} | {m['shuffled_conditioning_l1']:.4f} | - | - |")
            continue
        f, m = r["final"], r["final_monitor"]
        L.append(f"| {a} | {f['l1']:.4f} | {f['per_crop_median']:.4f} / {f['per_crop_p90']:.4f} / {f['per_crop_max']:.4f} | {f['anchor_l1']:.4f} | "
                 f"{f['l1_interior']:.4f} / {f['l1_boundary']:.4f} | {f['time_diff_l1']:.4f} / {f['freq_diff_l1']:.4f} | "
                 f"{f.get('shuffled_conditioning_l1', float('nan')):.4f} | {m['l1']:.4f} | {m['shuffled_conditioning_l1']:.4f} | "
                 f"{r['runtime_hours'] * 60:.1f} min | {r['updates_per_second_median']:.0f} |")
    L += ["", "## Paired per-crop differences (b - a; negative = lower error; bootstrap 95% CI over crops)", "",
          "| comparison | mean [CI] | fraction lower |", "| --- | --- | --- |"]
    for k, c in C.items():
        L.append(f"| {k} | {c['mean']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {100 * c['fraction_b_better']:.0f}% |")
    L += ["", "## Secondary: best-training-L1 checkpoints", "", "| arm | best fit L1 | at presentations |", "| --- | --- | --- |"]
    for a, r in R.items():
        L.append(f"| {a} | {r['best']['fit_l1']:.4f} | {r['best']['presentations']:,} |")
    lb = R["B"]["final_train_log"] or {}
    L += ["", "## Discriminator (arm B)", "",
          (f"- Warm-up (100 D-only updates): loss {R['B']['d_warmup']['first']['d_loss']:.3f} -> {R['B']['d_warmup']['last']['d_loss']:.3f}, "
          f"accuracy {R['B']['d_warmup']['last']['d_accuracy']:.3f}."),
          (f"- Last training window: D loss {lb.get('d_loss_window_mean', float('nan')):.3f}, real/fake logits "
          f"{lb.get('d_real_logit_mean_window_mean', float('nan')):+.3f} / {lb.get('d_fake_logit_mean_window_mean', float('nan')):+.3f}, accuracy "
          f"{lb.get('d_accuracy_window_mean', float('nan')):.3f}, weighted adv / L1 gradient {lb.get('weighted_adv_over_l1_grad') or float('nan'):.3f}."),
          f"- Non-finite events: A {R['A']['nonfinite_events']}, B {R['B']['nonfinite_events']}; spikes: A {len(R['A']['spikes'])}, B {len(R['B']['spikes'])}.",
          "", "## Compute (equal generator exposure, unequal total compute)", "",
          (f"- A ≈ {rep['compute']['A_train_macs_estimate'] / 1e12:.0f} TMAC, B ≈ {rep['compute']['B_train_macs_estimate'] / 1e12:.0f} TMAC (analytic). "
          f"Measured: A {R['A']['runtime_hours'] * 60:.1f} min, B {R['B']['runtime_hours'] * 60:.1f} min; billing-hours A {R['A']['allocation_billing_hours']:.2f}, "
          f"B {R['B']['allocation_billing_hours']:.2f}."),
          "", "## Listening", "",
          (f"Blinded package `{rep['listening']['package']}` (key `{rep['listening']['key']}`): 8 training + 8 monitoring examples fixed before training, "
          "clips start / A final / B final in random order, common gain. **Ratings pending.**"), "",
          "## Interpretation limits", "",
          "- Discriminator scores and sharper spectrograms are not evidence of fidelity; mel L1 to the source and listening are.",
          "- Training-pool changes are fitting effects; monitoring changes on 16 reused crops are exploratory transfer evidence.",
          "- Equal generator exposure is not equal compute: B trains a discriminator and runs extra backward passes.",
          ("- An adversarial gain would not show recovery of previously inaccessible information; a null result applies only to this "
          "discriminator, weighting, pool and budget. Single seed."), "",
          "Plots: `curves.png`, `adversarial_dynamics.png`, `paired_per_crop.png`; spectrograms (shared scales, transient and harmonic zooms) in `spectrograms/`."]
    atomic_write_text(art / "TASK-27_RESULTS.md", "\n".join(L) + "\n")


def score_listening(art: Path) -> None:
    blind, key = art / "listening_blind", json.loads((art / "listening_blind_KEY" / "key.json").read_text())
    with open(blind / "ratings.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    with open(blind / "preferences.csv", newline="") as f:
        prefs = list(csv.DictReader(f))
    scales = ("quality_artifacts_1to5", "transient_sharpness_1to5", "pitch_timbre_fidelity_1to5")
    missing = [r for r in rows if any(not r[s].strip() for s in scales)]
    if missing:
        raise SystemExit(f"{len(missing)} clip rows are not fully rated; nothing is scored (ratings are never filled in automatically)")
    out = {"per_condition": {}, "pairwise_B_vs_A": {}, "preference": {}}
    for grp in ("training", "monitoring", "all"):
        sel = [r for r in rows if grp == "all" or key["examples"][r["example"]]["group"] == grp]
        per = {c: {s: [float(r[s]) for r in sel if key["examples"][r["example"]][r["clip"]] == c] for s in scales} for c in CONDITIONS}
        out["per_condition"][grp] = {c: {s: float(np.mean(v)) for s, v in d.items()} for c, d in per.items()}
        pw = {}
        for s in scales:
            diffs = []
            for e in {r["example"] for r in sel}:
                vals = {key["examples"][e][r["clip"]]: float(r[s]) for r in sel if r["example"] == e}
                diffs.append(vals["B_final"] - vals["A_final"])
            pw[s] = {"B_better": sum(d > 0 for d in diffs), "A_better": sum(d < 0 for d in diffs), "ties": sum(d == 0 for d in diffs)}
        out["pairwise_B_vs_A"][grp] = pw
    out["preference"] = {c: sum(1 for p in prefs if p["preferred_clip_overall"] in key["examples"][p["example"]]
                                and key["examples"][p["example"]][p["preferred_clip_overall"]] == c) for c in CONDITIONS}
    atomic_write_text(art / "listening_scores.json", json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


# --- main --------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "calibrate", "train", "report", "all", "score-listening"])
    parser.add_argument("--config", type=Path, default=Path("configs/task27.yaml"))
    parser.add_argument("--device", choices=DEVICE_CHOICES, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--arms", default=None, help="comma-separated arm names (default: A,B)")
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
    if args.stage == "score-listening":
        score_listening(art)
        return 0
    if args.stage in ("report", "all") and (art / "report.json").exists():
        raise SystemExit(f"{art / 'report.json'} exists: the report of {args.run_name} is never overwritten")
    device = select_device(args.device or cfg["device_request"], cfg["allow_cpu_fallback"])
    require_fp32_default()
    dev_info = device_metadata(args.device or cfg["device_request"], device)
    env = capture_environment_manifest(REPO_ROOT)
    print(f"TASK-27 {args.stage} {args.run_name} | device {json.dumps(dev_info)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    mpath = run_dir / "run_manifest.json"
    if not mpath.exists():
        try:
            fd = os.open(mpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({"task": "TASK-27", "run_name": args.run_name, "config": cfg, "recipe": recipe,
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
        prepared = prepare(cfg, recipe, device, run_dir)
    elif not (run_dir / "prepared.json").exists():
        raise SystemExit("run the prepare stage first")
    else:
        prepared = json.loads((run_dir / "prepared.json").read_text())
    if args.stage in ("calibrate", "all"):
        seed_everything(cfg["seed"], device)
        calibrate(cfg, recipe, device, run_dir, prepared)
    if args.stage in ("train", "all"):
        arms = args.arms.split(",") if args.arms else list(cfg["arms"])
        bad = set(arms) - set(cfg["arms"])
        if bad:
            raise SystemExit(f"unknown arms {sorted(bad)}")
        sa = (args.stop_after.split(":")[0], int(args.stop_after.split(":")[1])) if args.stop_after else None
        code = train_arms(cfg, recipe, device, run_dir, prepared, arms, sa, args.steal_lock, stop, env, args.wandb_mode)
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
