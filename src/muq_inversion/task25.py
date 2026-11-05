"""TASK-25: architecture-versus-optimization ablation on TASK-24's fixed N=256 fitting pool
(helpers; the driver is scripts/task25_ablation.py).

Arms: A/B/C/D = current decoder at batch 1 or 16 and initial LR 1e-3 or 1e-4; E = width x2; F = learned
temporal resampler. Every arm sees exactly 819,200 crop presentations from the same seeded shuffled
sample stream (grouped into batches), with an LR schedule indexed by cumulative presentations.

F's resampler (`PhaseResampler`): for mel frame j at fractional MuQ position p_j = lo_j + w_j (frame-
center alignment, `time_alignment.source_positions`; 3.75 mel frames per MuQ frame, so the phase w
cycles through 15 values), it combines the 4 neighbouring projected MuQ frames lo-1, lo, lo+1, lo+2
(indices clamped to [0, T-1], i.e. edge replication as in the linear interpolation) with per-channel
weights base(w) + delta(w): base = [0, 1-w, w, 0] (exactly the existing linear interpolation) and delta
= MLP(w) (1 -> 32 -> GELU -> 4*C), whose last layer is zero-initialized, so at initialization F is
bit-for-bit the current decoder. The rule is shared by all crops and positions (it depends only on
the phase), so it is local and translation-equivariant, with no absolute-position table.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from muq_inversion.models.muq_mel_translator import MuQToMelTranslator
from muq_inversion.task23 import (
    NonFiniteError,
    _rng_states,
    _set_rng_states,
    atomic_torch_save,
    masked_l1,
    state_dict_sha256,
)
from muq_inversion.task24 import ShuffledPassSampler, evaluate_set

TAP_OFFSETS = (-1, 0, 1, 2)


class PhaseResampler(nn.Module):
    """Learned local resampler: (B, C, T_src) at fractional positions (T_tgt,) -> (B, C, T_tgt)."""

    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.channels = channels
        self.mlp_in = nn.Linear(1, hidden)
        self.mlp_out = nn.Linear(hidden, len(TAP_OFFSETS) * channels)
        nn.init.zeros_(self.mlp_out.weight)
        nn.init.zeros_(self.mlp_out.bias)

    def taps(self, positions: torch.Tensor, t_src: int) -> tuple[torch.Tensor, torch.Tensor]:
        """positions (T_tgt,) -> tap indices (K, T_tgt) and phases w (T_tgt,), as interpolate_at_positions."""
        if positions.min() < 0 or positions.max() > t_src - 1:
            raise ValueError(f"positions must lie in [0, {t_src - 1}]")
        lo = positions.floor().long().clamp(max=t_src - 1)
        w = positions - lo.to(positions.dtype)
        idx = torch.stack([(lo + k).clamp(0, t_src - 1) for k in TAP_OFFSETS])
        return idx, w

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        pos = positions.to(x.device, x.dtype)
        idx, w = self.taps(pos, x.shape[-1])
        delta = self.mlp_out(F.gelu(self.mlp_in(w[:, None]))).view(-1, len(TAP_OFFSETS), self.channels)  # (T, K, C)
        base = torch.stack([torch.zeros_like(w), 1 - w, w, torch.zeros_like(w)])  # (K, T)
        out = None
        for k in (1, 2, 0, 3):  # (1-w)*x[lo] + w*x[hi] first: at init this equals the linear interpolation exactly
            wk = base[k][None, :] + delta[:, k, :].transpose(0, 1)  # (C, T)
            term = x[..., idx[k]] * wk
            out = term if out is None else out + term
        return out

    @property
    def support_source_frames(self) -> int:
        return len(TAP_OFFSETS)


class ResampledMuQToMelTranslator(MuQToMelTranslator):
    """Arm F: the current decoder (same projection, TemporalConvNet trunk, output head, width) with the
    fixed linear interpolation replaced by `PhaseResampler`. Requires frame-center source positions."""

    def __init__(self, input_channels: int = 1024, hidden_channels: int = 256, num_mels: int = 100, resampler_hidden: int = 32):
        super().__init__(input_channels, hidden_channels, num_mels)
        self.resampler = PhaseResampler(hidden_channels, resampler_hidden)

    def forward(self, muq_features: torch.Tensor, num_mel_frames: int, source_positions: torch.Tensor | None = None) -> torch.Tensor:
        """muq_features (B, T_muq, C_in) -> log-mel (B, num_mels, num_mel_frames)."""
        if source_positions is None or source_positions.shape != (num_mel_frames,):
            raise ValueError("ResampledMuQToMelTranslator needs (num_mel_frames,) frame-center source positions")
        h = self.input_projection(muq_features).transpose(1, 2)
        h = self.resampler(h, source_positions)
        h = self.temporal_conv(h.transpose(1, 2))
        return self.output_projection(h).transpose(1, 2)


def build_model(arch: str, input_channels: int, hidden: int, num_mels: int, width_multiplier: int = 1,
                resampler_hidden: int = 32) -> nn.Module:
    if arch == "current":
        return MuQToMelTranslator(input_channels, hidden * width_multiplier, num_mels)
    if arch == "learned_resampler":
        return ResampledMuQToMelTranslator(input_channels, hidden, num_mels, resampler_hidden)
    raise ValueError(f"unknown architecture {arch}")


def forward_macs(model: nn.Module, t_muq: int, t_mel: int) -> dict:
    """Analytic multiply-accumulates per crop forward pass (per-layer breakdown)."""
    c_in, h = model.input_projection.in_features, model.input_projection.out_features
    k, m = model.temporal_conv.conv1.kernel_size[0], model.output_projection.out_features
    out = {"input_projection": t_muq * c_in * h, "temporal_conv": 3 * t_mel * h * h * k, "output_projection": t_mel * h * m}
    if isinstance(model, ResampledMuQToMelTranslator):
        r = model.resampler
        out["resampler"] = t_mel * (r.mlp_in.out_features + r.mlp_in.out_features * r.mlp_out.out_features + len(TAP_OFFSETS) * h)
    else:
        out["interpolation"] = t_mel * 2 * h
    out["total"] = sum(out.values())
    return out


def source_receptive_field(model: nn.Module, positions: torch.Tensor, t_muq: int) -> dict:
    """MuQ frames that can influence each output mel frame (trunk receptive field in mel frames, then
    the resampler/interpolation support): min/median/max count over frames."""
    rf = model.temporal_conv.receptive_field_frames
    half = (rf - 1) // 2
    p = positions.detach().cpu()
    lo = p.floor().long().clamp(max=t_muq - 1)
    offs = TAP_OFFSETS if isinstance(model, ResampledMuQToMelTranslator) else (0, 1)
    taps = [{(lo[j] + o).clamp(0, t_muq - 1).item() for o in offs} for j in range(len(p))]
    counts = [len(set().union(*taps[max(0, j - half):j + half + 1])) for j in range(len(p))]
    return {"trunk_receptive_field_mel_frames": rf, "resampling_support_source_frames": len(offs),
            "source_frames_per_output_min_median_max": [min(counts), int(np.median(counts)), max(counts)]}


# --- exposure-based training --------------------------------------------------------------


def lr_at_presentations(presented_before: int, base: float, drops: tuple[tuple[int, float], ...]) -> float:
    """LR for an update that starts after `presented_before` crop presentations: `base`, times each
    factor once that many presentations have been reached (drops ((204800, 0.1), (614400, 0.1)))."""
    lr = base
    for at, f in drops:
        if presented_before >= at:
            lr *= f
    return lr


@dataclass
class ExposureSpec:
    name: str
    batch: int
    base_lr: float
    betas: tuple[float, float]
    total_presentations: int
    lr_drops: tuple[tuple[int, float], ...]
    log_every: int = 1000
    fit_eval_every: int = 16384
    monitor_eval_every: int = 65536
    latest_every: int = 65536
    matched_updates: tuple[int, ...] = (10000, 50000)
    chunk: int = 64
    sampler_seed: str = "task25-sampler-0"
    spike_factor: float = 2.0
    recovery_factor: float = 1.1
    anchor_index: int = 0
    showcase: tuple[int, ...] = (0, 1, 2, 3)

    @property
    def updates(self) -> int:
        if self.total_presentations % self.batch:
            raise ValueError("the budget must be a whole number of batches")
        return self.total_presentations // self.batch

    def boundary_updates(self) -> dict[int, int]:
        """{presentations: update} just before each LR reduction."""
        return {at: at // self.batch for at, _ in self.lr_drops if at % self.batch == 0}

    def special_updates(self) -> set[int]:
        return set(self.boundary_updates().values()) | {u for u in self.matched_updates if u <= self.updates} | {self.updates}


def train_exposure_arm(spec: ExposureSpec, model: nn.Module, data: dict, run_dir: Path, device: torch.device,
                       meta: dict | None = None, stop_after_update: int | None = None,
                       should_stop: Callable[[], bool] = lambda: False, log: Callable[[str], None] = print,
                       metrics_sink: Callable[[str, dict, int], None] | None = None,
                       state_hook: Callable[[dict], None] | None = None) -> dict | None:
    """Train one arm resumably for exactly spec.total_presentations crop presentations.

    data: fit_feats (N, T_muq, C), fit_mel (N, M, T), mon_feats, mon_mel, mask, boundary, positions.
    Batches are consecutive groups of `spec.batch` draws from one ShuffledPassSampler stream (the same
    seed in every arm). Outputs as TASK-24 (train_log / fit_eval / monitor_eval JSONL, checkpoints
    init/best/final/latest/lr_boundary_*, preds, summary.json). `metrics_sink(kind, row, presentations)`
    receives every logged row (W&B); `state_hook(state)` may add fields (e.g. the W&B run id) before
    the first checkpoint."""
    run_dir = Path(run_dir)
    ck, pr = run_dir / "checkpoints", run_dir / "preds"
    latest, summary_path = ck / "latest.pt", run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    fit_feats, fit_mel, mon_feats, mon_mel = (data[k] for k in ("fit_feats", "fit_mel", "mon_feats", "mon_mel"))
    mask, boundary, positions = data["mask"], data["boundary"], data["positions"]
    n = fit_feats.shape[0]
    t_mel = fit_mel.shape[-1]
    mask_dev = mask.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=spec.base_lr, betas=tuple(spec.betas), weight_decay=0.0)
    sampler = ShuffledPassSampler(n, spec.sampler_seed)
    specials, boundaries = spec.special_updates(), spec.boundary_updates()
    files = {k: run_dir / f"{k}.jsonl" for k in ("train_log", "fit_eval", "monitor_eval")}

    def write(kind: str, row: dict) -> None:
        with open(files[kind], "a") as f:
            f.write(json.dumps(row) + "\n")
        if metrics_sink is not None:
            metrics_sink(kind, row, row["presentations"])

    def checkpoint(path: Path) -> None:
        atomic_torch_save(path, {"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                                 "sampler_state": sampler.state_dict(), "rng": _rng_states(device), "state": state,
                                 "spec": asdict(spec), "meta": meta or {}})
        back = torch.load(path, map_location="cpu", weights_only=False)
        assert back["state"]["update"] == state["update"] and back["sampler_state"]["counts"] == sampler.counts, f"{path}: validation failed"
        k0 = next(iter(back["model_state_dict"]))
        assert torch.equal(back["model_state_dict"][k0], model.state_dict()[k0].cpu()), f"{path}: tensor mismatch"

    def wall() -> float:
        return state["runtime_seconds"] + time.perf_counter() - t0

    def evaluate(update: int, monitor: bool) -> tuple[dict, torch.Tensor, torch.Tensor | None]:
        p = update * spec.batch
        fm, per_crop, fp = evaluate_set(model, fit_feats, fit_mel, mask, boundary, positions, device, spec.chunk)
        row = {"presentations": p, "update": update, "presentations_per_crop": p / n, "lr": state["lr"], "wall_seconds": wall(),
               **fm, "anchor_l1": per_crop[spec.anchor_index], "per_crop_l1": per_crop}
        write("fit_eval", row)
        mp = None
        if monitor:
            mm, mpc, mp = evaluate_set(model, mon_feats, mon_mel, mask, boundary, positions, device, spec.chunk)
            write("monitor_eval", {"presentations": p, "update": update, "lr": state["lr"], "wall_seconds": wall(), **mm, "per_crop_l1": mpc})
            row["monitor"] = mm
        sp = state["spike"]
        if sp["open"] is None and update > 0 and fm["l1"] > spec.spike_factor * state["best"]["fit_l1"]:
            sp["open"] = {"presentations": p, "update": update, "fit_l1": fm["l1"], "pre_spike_best": state["best"]["fit_l1"]}
            log(f"[{spec.name}] spike at {p:,} presentations: fit L1 {fm['l1']:.4f} vs best {state['best']['fit_l1']:.4f}")
        elif sp["open"] is not None:
            sp["open"]["peak_fit_l1"] = max(sp["open"].get("peak_fit_l1", sp["open"]["fit_l1"]), fm["l1"])
            if fm["l1"] <= spec.recovery_factor * sp["open"]["pre_spike_best"]:
                sp["events"].append({**sp["open"], "recovered_at_presentations": p,
                                     "recovery_presentations": p - sp["open"]["presentations"]})
                sp["open"] = None
        if fm["l1"] < state["best"]["fit_l1"]:
            state["best"] = {"presentations": p, "update": update, "fit_l1": fm["l1"], "anchor_l1": per_crop[spec.anchor_index],
                             "per_crop_l1": per_crop}
            atomic_torch_save(ck / "best.pt", {"model_state_dict": model.state_dict(), "update": update, "presentations": p,
                                               "fit_l1": fm["l1"], "spec": asdict(spec), "meta": meta or {}})
            atomic_torch_save(pr / "best.pt", {"update": update, "presentations": p, "showcase": list(spec.showcase), "fit": fp[list(spec.showcase)]})
        return row, fp, mp

    if latest.exists():
        c = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model_state_dict"])
        opt.load_state_dict(c["optimizer_state_dict"])
        sampler.load_state_dict(c["sampler_state"])
        _set_rng_states(c["rng"], device)
        state = c["state"]
        for f in files.values():
            truncate_jsonl_presentations(f, state["update"] * spec.batch)
        state["lineage"].append({"resumed_at_update": state["update"], "utc": datetime.now(UTC).isoformat(),
                                 "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        t0 = time.perf_counter()
        if state_hook:
            state_hook(state)
        log(f"[{spec.name}] resumed at update {state['update']:,} ({state['update'] * spec.batch:,} presentations)")
    else:
        if run_dir.exists() and any(p.name != "LOCK" for p in run_dir.iterdir()):
            raise SystemExit(f"{run_dir} has files but no latest.pt; refusing to overwrite")
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {"update": 0, "lr": spec.base_lr, "best": {"update": None, "fit_l1": float("inf")}, "spike": {"open": None, "events": []},
                 "runtime_seconds": 0.0, "nonfinite_events": 0, "window": {"n": 0, "sum": 0.0, "max": 0.0, "grad_max": 0.0},
                 "init_sha256": state_dict_sha256(model),
                 "lineage": [{"started_at_update": 0, "utc": datetime.now(UTC).isoformat(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}]}
        t0 = time.perf_counter()
        if state_hook:
            state_hook(state)
        atomic_torch_save(ck / "init.pt", {"model_state_dict": model.state_dict(), "sha256": state["init_sha256"], "spec": asdict(spec)})
        row, fp, mp = evaluate(0, monitor=True)
        atomic_torch_save(pr / "update_000000.pt", {"update": 0, "showcase": list(spec.showcase), "fit": fp[list(spec.showcase)], "monitor": mp})
        checkpoint(latest)
    params = [p for p in model.parameters() if p.requires_grad]
    w = state["window"]
    t_log, u_log = time.perf_counter(), state["update"]
    while state["update"] < spec.updates:
        u = state["update"] + 1
        before = (u - 1) * spec.batch
        idx = torch.tensor([sampler.next() for _ in range(spec.batch)], device=fit_feats.device)
        lr = lr_at_presentations(before, spec.base_lr, spec.lr_drops)
        model.train()
        loss = masked_l1(model(fit_feats[idx], t_mel, positions), fit_mel[idx], mask_dev)
        if not torch.isfinite(loss):
            state["nonfinite_events"] += 1
            checkpoint(ck / f"nonfinite_{u:07d}.pt")
            raise NonFiniteError(f"[{spec.name}] non-finite loss at update {u}; state saved")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, float("inf")))  # measure only: no clipping, as TASK-22/23/24
        if not np.isfinite(gn):
            state["nonfinite_events"] += 1
            checkpoint(ck / f"nonfinite_{u:07d}.pt")
            raise NonFiniteError(f"[{spec.name}] non-finite gradient norm at update {u}; state saved")
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        state["update"], state["lr"] = u, lr
        p = u * spec.batch
        lv = loss.item()
        w["n"], w["sum"], w["max"], w["grad_max"] = w["n"] + 1, w["sum"] + lv, max(w["max"], lv), max(w["grad_max"], gn)
        if p // spec.log_every > before // spec.log_every:
            now = time.perf_counter()
            c = np.array(sampler.counts)
            ups = (u - u_log) / max(now - t_log, 1e-9)
            write("train_log", {"presentations": p, "update": u, "lr": lr, "train_loss_window_mean": w["sum"] / w["n"],
                                "train_loss_window_max": w["max"], "grad_norm": gn, "grad_norm_window_max": w["grad_max"],
                                "updates_per_second": ups, "presentations_per_second": ups * spec.batch, "wall_seconds": wall(),
                                "presentations_min": int(c.min()), "presentations_max": int(c.max())})
            w.update(n=0, sum=0.0, max=0.0, grad_max=0.0)
            t_log, u_log = now, u
        if p % spec.fit_eval_every == 0 or u in specials:
            row, fp, mp = evaluate(u, monitor=p % spec.monitor_eval_every == 0 or u in specials)
            if p % (spec.fit_eval_every * 10) == 0 or u in specials:
                log(f"[{spec.name}] {p:,} presentations (update {u:,}): fit L1 {row['l1']:.5f} (median {row['per_crop_median']:.5f}, "
                    f"max {row['per_crop_max']:.5f}) anchor {row['anchor_l1']:.5f}"
                    + (f" | monitor {row['monitor']['l1']:.4f}" if "monitor" in row else "") + f" | lr {lr:g}")
            if u in specials:
                atomic_torch_save(pr / f"update_{u:07d}.pt", {"update": u, "presentations": p, "showcase": list(spec.showcase),
                                                              "fit": fp[list(spec.showcase)], "monitor": mp})
            if u == spec.updates:
                atomic_torch_save(pr / "final.pt", {"update": u, "presentations": p, "fit": fp, "monitor": mp})
        for at, bu in boundaries.items():
            if u == bu:
                checkpoint(ck / f"lr_boundary_{at:07d}.pt")
        interrupted = should_stop() or u == stop_after_update
        if p % spec.latest_every == 0 or interrupted or u == spec.updates:
            state["runtime_seconds"] = wall()
            t0 = time.perf_counter()
            checkpoint(latest)
        if interrupted:
            log(f"[{spec.name}] interrupted at update {u:,}; latest.pt saved")
            return None
    checkpoint(ck / "final.pt")
    rows = {k: [json.loads(x) for x in f.read_text().splitlines()] for k, f in files.items()}
    at_u = {r["update"]: r for r in rows["fit_eval"]}
    mon_u = {r["update"]: r for r in rows["monitor_eval"]}

    def brief(r: dict | None) -> dict | None:
        return None if r is None else {k: v for k, v in r.items() if k not in ("per_crop_l1", "monitor")}

    tl = rows["train_log"]
    summary = {"arm": spec.name, "spec": asdict(spec), "meta": meta or {}, "init_sha256": state["init_sha256"],
               "final_sha256": state_dict_sha256(model), "updates_run": state["update"], "presentations": state["update"] * spec.batch,
               "initial": brief(at_u[0]), "final": brief(at_u[spec.updates]), "final_monitor": brief(mon_u.get(spec.updates)),
               "best": state["best"],
               "lr_boundaries": {at: {"update": bu, "fit": brief(at_u.get(bu)), "monitor": brief(mon_u.get(bu))} for at, bu in boundaries.items()},
               "matched_updates": {u: {"presentations": u * spec.batch, "fit": brief(at_u.get(u)), "monitor": brief(mon_u.get(u)),
                                       "lr": lr_at_presentations((u - 1) * spec.batch, spec.base_lr, spec.lr_drops)}
                                   for u in spec.matched_updates if u <= spec.updates},
               "spikes": state["spike"]["events"] + ([state["spike"]["open"] | {"recovered_at_presentations": None}] if state["spike"]["open"] else []),
               "presentation_counts_min_max": [min(sampler.counts), max(sampler.counts)], "passes": sampler.passes,
               "nonfinite_events": state["nonfinite_events"], "runtime_seconds": state["runtime_seconds"],
               "updates_per_second_median": float(np.median([r["updates_per_second"] for r in tl])) if tl else None,
               "grad_norm_window_max_overall": max(r["grad_norm_window_max"] for r in tl) if tl else None,
               "lineage": state["lineage"], "wandb_run_id": state.get("wandb_run_id")}
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def truncate_jsonl_presentations(path: Path, max_presentations: int) -> None:
    """Drop rows logged after the checkpoint being resumed (they are logged again)."""
    if path.exists():
        rows = [line for line in path.read_text().splitlines() if json.loads(line)["presentations"] <= max_presentations]
        path.write_text("".join(r + "\n" for r in rows))


def paired_difference(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Per-crop paired difference b - a (negative = b fits better): mean, median, bootstrap 95% CI over crops."""
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"mean": float(d.mean()), "median": float(np.median(d)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "fraction_b_better": float((d < 0).mean()), "n": len(d)}

