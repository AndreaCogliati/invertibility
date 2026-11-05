"""TASK-24: fixed-crop decoder fitting ladder (helpers; the driver is scripts/task24_ladder.py).

Nested fitting subsets N = 1, 4, 16, 64, 256 of TASK-22 training crops (first-block MuQ features ->
BigVGAN-v2 log-mel), N=1 being the exact TASK-23 pair. Every arm starts from TASK-23 B's initial
weights, trains with batch size one on seeded shuffled passes (no replacement within a pass) for a
fixed budget and a fixed LR schedule, and is evaluated on its WHOLE fitting subset plus a separate
16-crop monitoring set of training tracks outside the 256-crop pool (monitoring only: it never
selects checkpoints, changes the schedule, or stops training).

Reused from TASK-23 (`task23`): masked L1 and detail metrics, the boundary definition, RNG
preservation, atomic saves, tensor and state-dict hashes.
"""

from __future__ import annotations

import json
import os
import random
import socket
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from muq_inversion.task23 import (
    NonFiniteError,
    _rng_states,
    _set_rng_states,
    atomic_torch_save,
    masked_diff_errors,
    masked_l1,
    preserved_rng,
    state_dict_sha256,
)

# --- nested selection --------------------------------------------------------------------


def select_ladder(pool_rows: list[dict], anchor_crop_id: str, n_fit: int, n_monitor: int, seed: int,
                  available: set[tuple[str, int]] | None = None) -> tuple[list[dict], list[dict]]:
    """Deterministic nested order of fitting crops plus a disjoint monitoring set.

    pool_rows: TASK-22 training-pool rows (role train_pool, split train). The anchor comes first.
    Tracks are interleaved round-robin over genres (each genre's tracks and the genre order shuffled
    with `seed`; the anchor's genre last in each round, since round one already has it), one crop per
    track chosen with the same RNG, so the first N entries are the N-crop subset for every N and
    distinct tracks are used while they last. If tracks run out (only in small smoke caches), further
    rounds take additional crops of already-used tracks. The monitoring set continues the same
    track order with tracks NOT used by any fitting crop. Nothing depends on decoder performance.
    `available` (smoke only) restricts candidates to crops present in a cache."""
    rng = random.Random(f"task24-ladder-{seed}")
    rows = [r for r in pool_rows if available is None or (r["path"], int(r["crop_start_sample"])) in available]
    anchor = next((r for r in pool_rows if r["crop_id"] == anchor_crop_id), None)
    if anchor is None:
        raise ValueError(f"anchor {anchor_crop_id} is not a training-pool crop")
    by_track: dict[str, list[dict]] = {}
    for r in rows:
        if r["crop_id"] != anchor_crop_id:
            by_track.setdefault(r["track_id"], []).append(r)
    for v in by_track.values():
        v.sort(key=lambda r: int(r["crop_start_sample"]))
    genres: dict[str, list[str]] = {}
    for t in sorted(by_track):
        genres.setdefault(by_track[t][0]["genre"], []).append(t)
    for g in sorted(genres):
        rng.shuffle(genres[g])
    order = sorted(genres)
    rng.shuffle(order)
    order = [g for g in order if g != anchor["genre"]] + ([anchor["genre"]] if anchor["genre"] in genres else [])
    queue = {g: list(genres[g]) for g in order}
    track_seq: list[str] = []
    while any(queue.values()):
        for g in order:
            if queue[g]:
                track_seq.append(queue[g].pop(0))
    anchor_track_first = [t for t in track_seq if t != anchor["track_id"]]
    fit = [anchor]
    used_tracks = {anchor["track_id"]}
    remaining = {t: list(by_track[t]) for t in anchor_track_first}
    if anchor["track_id"] in by_track:  # the anchor's own track: other crops only as a last resort
        remaining[anchor["track_id"]] = list(by_track[anchor["track_id"]])
    for t in anchor_track_first:
        if len(fit) >= n_fit:
            break
        fit.append(remaining[t].pop(rng.randrange(len(remaining[t]))))
        used_tracks.add(t)
    while len(fit) < n_fit:  # small caches only: more crops of tracks already in the pool
        progress = False
        for t in anchor_track_first + [anchor["track_id"]]:
            if len(fit) >= n_fit:
                break
            if t in used_tracks and remaining.get(t):
                fit.append(remaining[t].pop(rng.randrange(len(remaining[t]))))
                progress = True
        if not progress:
            raise ValueError(f"only {len(fit)} candidate crops for N={n_fit}")
    mon_tracks = [t for t in anchor_track_first if t not in used_tracks][:n_monitor]
    if len(mon_tracks) < n_monitor:
        raise ValueError(f"only {len(mon_tracks)} unused tracks for a {n_monitor}-crop monitoring set")
    monitor = [by_track[t][rng.randrange(len(by_track[t]))] for t in mon_tracks]
    return fit, monitor


def anchor_row_matches(cache_tensors: dict, i: int, pair_tensors: dict) -> bool:
    """True if TASK-22 cache row i is bit-for-bit TASK-23's cached pair (features, target mel, waveform)."""
    return all(torch.equal(cache_tensors[k][i], v) for k, v in (("muq", pair_tensors["features"][0]),
                                                                ("mel", pair_tensors["target_mel"][0]),
                                                                ("wave", pair_tensors["waveform"])))


# --- sampler and schedule ----------------------------------------------------------------


class ShuffledPassSampler:
    """Seeded shuffled passes through range(n), without replacement within a pass; exact
    per-index presentation counts; resumable state."""

    def __init__(self, n: int, seed: str):
        self.n, self.rng = n, random.Random(seed)
        self.perm: list[int] = []
        self.pos, self.passes = 0, 0
        self.counts = [0] * n

    def next(self) -> int:
        if self.pos == len(self.perm):
            self.perm = list(range(self.n))
            self.rng.shuffle(self.perm)
            self.pos, self.passes = 0, self.passes + 1
        i = self.perm[self.pos]
        self.pos += 1
        self.counts[i] += 1
        return i

    def state_dict(self) -> dict:
        return {"n": self.n, "rng": self.rng.getstate(), "perm": list(self.perm), "pos": self.pos,
                "passes": self.passes, "counts": list(self.counts)}

    def load_state_dict(self, s: dict) -> None:
        if s["n"] != self.n:
            raise ValueError(f"sampler state is for n={s['n']}, not {self.n}")
        self.rng.setstate(s["rng"])
        self.perm, self.pos, self.passes, self.counts = list(s["perm"]), s["pos"], s["passes"], list(s["counts"])


def lr_at(step: int, base: float, drops: tuple[tuple[int, float], ...]) -> float:
    """LR used by update `step` (1-based): `base`, multiplied by each factor for updates AFTER its
    step. With drops ((50000, 0.1), (150000, 0.1)): updates 1..50000 at base, 50001..150000 at
    base/10, 150001.. at base/100. The checkpoint after update 50000 is just before the reduction."""
    lr = base
    for at, factor in drops:
        if step > at:
            lr *= factor
    return lr


# --- whole-set evaluation ----------------------------------------------------------------


def evaluate_set(model: nn.Module, feats: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                 boundary: torch.Tensor, positions: torch.Tensor, device: torch.device,
                 chunk: int = 64) -> tuple[dict, list[float], torch.Tensor]:
    """Evaluation-mode predictions for a whole set, without changing persistent model state or RNG.
    feats (n, T_muq, C) and targets (n, M, T) on any device; mask/boundary (T,) bool shared by all
    crops (fixed-length crops). Returns aggregate metrics (masked absolute error averaged over all
    valid mel elements of the set), per-crop L1, and the predictions (n, M, T) on CPU."""
    was = model.training
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t_mel = targets.shape[-1]
    with preserved_rng(device), torch.no_grad():
        model.eval()
        preds = torch.cat([model(feats[i:i + chunk].to(device), t_mel, positions).float().cpu()
                           for i in range(0, feats.shape[0], chunk)])
    model.train(was)
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    if changed:
        raise RuntimeError(f"evaluation changed persistent state: {changed}")
    tgt, m, b = targets.float().cpu(), mask.cpu(), boundary.cpu()
    w = m.float()[None, None, :]
    err = (preds - tgt).abs() * w
    per_crop = (err.sum((1, 2)) / (w.sum() * tgt.shape[1])).tolist()
    pc = np.array(per_crop)
    out = {"l1": float(masked_l1(preds, tgt, m)), "per_crop_mean": float(pc.mean()), "per_crop_median": float(np.median(pc)),
           "per_crop_p90": float(np.percentile(pc, 90)), "per_crop_max": float(pc.max()), "per_crop_argmax": int(pc.argmax()),
           "nonfinite_pred": int((~torch.isfinite(preds)).sum())}
    for name, sel in (("interior", m & ~b), ("boundary", m & b)):
        out[f"l1_{name}"] = float(masked_l1(preds, tgt, sel)) if bool(sel.any()) else float("nan")
    return out | masked_diff_errors(preds, tgt, m), per_crop, preds


# --- arm training ------------------------------------------------------------------------


@dataclass
class LadderSpec:
    n: int
    steps: int
    base_lr: float
    betas: tuple[float, float]
    lr_drops: tuple[tuple[int, float], ...]
    milestones: tuple[int, ...]
    exposures: tuple[int, ...]
    log_every: int = 100
    fit_eval_every: int = 1000
    monitor_eval_every: int = 5000
    latest_every: int = 5000
    chunk: int = 64
    sampler_seed: str = "task24-sampler-0"
    spike_factor: float = 2.0
    recovery_factor: float = 1.1
    anchor_index: int = 0
    showcase: tuple[int, ...] = (0,)

    def exposure_steps(self) -> dict[int, int]:
        """{presentations per crop: update count} for the reachable matched-exposure points."""
        return {k: k * self.n for k in self.exposures if k * self.n <= self.steps}

    def eval_steps(self) -> set[int]:
        return set(self.milestones) | set(self.exposure_steps().values()) | {self.steps}


class ArmLock:
    """Exclusive per-arm lock file (O_EXCL), so two processes never write the same arm. A stale lock
    (crashed job) must be removed explicitly (`steal=True`) after checking nothing is running."""

    def __init__(self, path: Path, steal: bool = False):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if steal and path.exists():
            path.unlink()
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(f"STOP: {path} exists ({path.read_text().strip()}): another writer may be running this arm. "
                             "If no job is running it, rerun with --steal-lock.") from None
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                "utc": datetime.now(UTC).isoformat()}))

    def release(self) -> None:
        if self.path.exists():
            self.path.unlink()


def truncate_jsonl(path: Path, max_step: int) -> None:
    """Drop rows logged after the checkpoint being resumed (they will be logged again)."""
    if path.exists():
        rows = [line for line in path.read_text().splitlines() if json.loads(line)["step"] <= max_step]
        path.write_text("".join(r + "\n" for r in rows))


def train_ladder_arm(spec: LadderSpec, model: nn.Module, data: dict, run_dir: Path, device: torch.device,
                     meta: dict | None = None, stop_after_step: int | None = None,
                     should_stop: Callable[[], bool] = lambda: False, log: Callable[[str], None] = print) -> dict | None:
    """Train one ladder arm, resumably. data: fit_feats (N, T_muq, C), fit_mel (N, M, T), mon_feats,
    mon_mel, mask (T,), boundary (T,), positions (T,) on `device`.

    run_dir: train_log.jsonl (every log_every updates), fit_eval.jsonl (whole fitting set every
    fit_eval_every and at every comparison step, with per-crop L1), monitor_eval.jsonl (every
    monitor_eval_every and at comparison steps), checkpoints/{milestone_XXXXXX, best, latest}.pt,
    preds/{step_XXXXXX, exposure_XXXx, best, final}.pt, summary.json. Returns the summary, or None
    when interrupted (latest.pt saved)."""
    run_dir = Path(run_dir)
    ck, pr = run_dir / "checkpoints", run_dir / "preds"
    latest, summary_path = ck / "latest.pt", run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    fit_feats, fit_mel, mon_feats, mon_mel = (data[k] for k in ("fit_feats", "fit_mel", "mon_feats", "mon_mel"))
    mask, boundary, positions = data["mask"], data["boundary"], data["positions"]
    if fit_feats.shape[0] != spec.n:
        raise ValueError(f"fitting set has {fit_feats.shape[0]} crops, spec N={spec.n}")
    if spec.steps not in spec.milestones:
        raise ValueError("the final step must be a milestone (it is the final checkpoint)")
    t_mel = fit_mel.shape[-1]
    mask_dev = mask.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=spec.base_lr, betas=tuple(spec.betas), weight_decay=0.0)
    sampler = ShuffledPassSampler(spec.n, spec.sampler_seed)
    exp_steps = spec.exposure_steps()
    eval_steps = spec.eval_steps()
    files = {k: run_dir / f"{k}.jsonl" for k in ("train_log", "fit_eval", "monitor_eval")}

    def write(kind: str, row: dict) -> None:
        with open(files[kind], "a") as f:
            f.write(json.dumps(row) + "\n")

    def checkpoint(path: Path) -> None:
        atomic_torch_save(path, {"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                                 "sampler_state": sampler.state_dict(), "rng": _rng_states(device), "state": state,
                                 "spec": asdict(spec), "meta": meta or {}})
        back = torch.load(path, map_location="cpu", weights_only=False)
        assert back["state"]["step"] == state["step"] and back["sampler_state"]["counts"] == sampler.counts, f"{path}: validation failed"
        k0 = next(iter(back["model_state_dict"]))
        assert torch.equal(back["model_state_dict"][k0], model.state_dict()[k0].cpu()), f"{path}: tensor mismatch"

    def evaluate(step: int, monitor: bool) -> tuple[dict, torch.Tensor, torch.Tensor | None]:
        fm, per_crop, fp = evaluate_set(model, fit_feats, fit_mel, mask, boundary, positions, device, spec.chunk)
        row = {"step": step, "lr": state["lr"], "presentations_per_crop": step / spec.n, **fm,
               "anchor_l1": per_crop[spec.anchor_index], "per_crop_l1": per_crop}
        write("fit_eval", row)
        mp = None
        if monitor:
            mm, mpc, mp = evaluate_set(model, mon_feats, mon_mel, mask, boundary, positions, device, spec.chunk)
            write("monitor_eval", {"step": step, "lr": state["lr"], **mm, "per_crop_l1": mpc})
            row["monitor"] = mm
        # spikes: whole-fitting-set L1 above spike_factor x the best so far; recovered below recovery_factor x
        sp = state["spike"]
        if sp["open"] is None and step > 0 and fm["l1"] > spec.spike_factor * state["best"]["fit_l1"]:
            sp["open"] = {"step": step, "fit_l1": fm["l1"], "pre_spike_best": state["best"]["fit_l1"]}
            log(f"[N={spec.n}] spike at {step}: fit L1 {fm['l1']:.4f} vs best {state['best']['fit_l1']:.4f}")
        elif sp["open"] is not None:
            sp["open"]["peak_fit_l1"] = max(sp["open"].get("peak_fit_l1", sp["open"]["fit_l1"]), fm["l1"])
            if fm["l1"] <= spec.recovery_factor * sp["open"]["pre_spike_best"]:
                sp["events"].append({**sp["open"], "recovered_at_step": step})
                sp["open"] = None
        if fm["l1"] < state["best"]["fit_l1"]:
            state["best"] = {"step": step, "fit_l1": fm["l1"], "anchor_l1": per_crop[spec.anchor_index],
                             "per_crop_l1": per_crop, "fit_metrics": fm}
            atomic_torch_save(ck / "best.pt", {"model_state_dict": model.state_dict(), "step": step, "fit_l1": fm["l1"],
                                               "spec": asdict(spec), "meta": meta or {}})
            atomic_torch_save(pr / "best.pt", {"step": step, "showcase": list(spec.showcase), "fit": fp[list(spec.showcase)]})
        return row, fp, mp

    if latest.exists():
        c = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model_state_dict"])
        opt.load_state_dict(c["optimizer_state_dict"])
        sampler.load_state_dict(c["sampler_state"])
        _set_rng_states(c["rng"], device)
        state = c["state"]
        for f in files.values():
            truncate_jsonl(f, state["step"])
        state["lineage"].append({"resumed_at_step": state["step"], "utc": datetime.now(UTC).isoformat(),
                                 "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        log(f"[N={spec.n}] resumed at step {state['step']}")
    else:
        if run_dir.exists() and any(p.name != "LOCK" for p in run_dir.iterdir()):
            raise SystemExit(f"{run_dir} has files but no latest.pt; refusing to overwrite")
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {"step": 0, "lr": spec.base_lr, "best": {"step": None, "fit_l1": float("inf")},
                 "spike": {"open": None, "events": []}, "runtime_seconds": 0.0, "nonfinite_events": 0,
                 "window": {"n": 0, "sum": 0.0, "max": 0.0, "grad_max": 0.0}, "init_sha256": state_dict_sha256(model),
                 "lineage": [{"started_at_step": 0, "utc": datetime.now(UTC).isoformat(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}]}
        row, fp, mp = evaluate(0, monitor=True)
        atomic_torch_save(pr / "step_000000.pt", {"step": 0, "showcase": list(spec.showcase), "fit": fp[list(spec.showcase)], "monitor": mp})
        checkpoint(latest)
    params = [p for p in model.parameters() if p.requires_grad]
    t0, t_log, s_log = time.perf_counter(), time.perf_counter(), state["step"]
    w = state["window"]
    while state["step"] < spec.steps:
        step = state["step"] + 1
        i = sampler.next()
        lr = lr_at(step, spec.base_lr, spec.lr_drops)
        model.train()
        loss = masked_l1(model(fit_feats[i:i + 1], t_mel, positions), fit_mel[i:i + 1], mask_dev)
        if not torch.isfinite(loss):
            state["nonfinite_events"] += 1
            checkpoint(ck / f"nonfinite_{step:06d}.pt")
            raise NonFiniteError(f"[N={spec.n}] non-finite loss at update {step} (crop {i}); state saved")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, float("inf")))  # measure only: TASK-22/23 never clip
        if not np.isfinite(gn):
            state["nonfinite_events"] += 1
            checkpoint(ck / f"nonfinite_{step:06d}.pt")
            raise NonFiniteError(f"[N={spec.n}] non-finite gradient norm at update {step} (crop {i}); state saved")
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        state["step"], state["lr"] = step, lr
        lv = loss.item()
        w["n"], w["sum"], w["max"], w["grad_max"] = w["n"] + 1, w["sum"] + lv, max(w["max"], lv), max(w["grad_max"], gn)
        if step % spec.log_every == 0:
            now = time.perf_counter()
            c = np.array(sampler.counts)
            write("train_log", {"step": step, "lr": lr, "train_loss_window_mean": w["sum"] / w["n"], "train_loss_window_max": w["max"],
                                "grad_norm": gn, "grad_norm_window_max": w["grad_max"], "updates_per_second": (step - s_log) / max(now - t_log, 1e-9),
                                "presentations_min": int(c.min()), "presentations_max": int(c.max()), "passes": sampler.passes})
            w.update(n=0, sum=0.0, max=0.0, grad_max=0.0)
            t_log, s_log = now, step
        full = step % spec.fit_eval_every == 0 or step in eval_steps
        if full:
            row, fp, mp = evaluate(step, monitor=step % spec.monitor_eval_every == 0 or step in eval_steps)
            if step % (spec.fit_eval_every * 10) == 0 or step in eval_steps:
                log(f"[N={spec.n}] step {step}: fit L1 {row['l1']:.5f} (median {row['per_crop_median']:.5f}, max {row['per_crop_max']:.5f}) "
                    f"anchor {row['anchor_l1']:.5f}" + (f" | monitor {row['monitor']['l1']:.4f}" if "monitor" in row else "") + f" | lr {lr:g}")
            for k, s in exp_steps.items():
                if step == s:
                    atomic_torch_save(pr / f"exposure_{k:04d}x.pt", {"step": step, "presentations_per_crop": k, "fit": fp, "monitor": mp,
                                                                     "presentation_counts": list(sampler.counts)})
            if step in spec.milestones:
                atomic_torch_save(pr / f"step_{step:06d}.pt", {"step": step, "showcase": list(spec.showcase), "fit": fp[list(spec.showcase)], "monitor": mp})
                if step == spec.steps:
                    atomic_torch_save(pr / "final.pt", {"step": step, "fit": fp, "monitor": mp})
        if step in spec.milestones:
            checkpoint(ck / f"milestone_{step:06d}.pt")
        interrupted = should_stop() or step == stop_after_step
        if step % spec.latest_every == 0 or interrupted or step == spec.steps:
            state["runtime_seconds"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            checkpoint(latest)
        if interrupted:
            log(f"[N={spec.n}] interrupted at update {step}; latest.pt saved")
            return None
    fit_rows = [json.loads(x) for x in files["fit_eval"].read_text().splitlines()]
    mon_rows = [json.loads(x) for x in files["monitor_eval"].read_text().splitlines()]
    at = {r["step"]: r for r in fit_rows}
    mon_at = {r["step"]: r for r in mon_rows}
    tl = [json.loads(x) for x in files["train_log"].read_text().splitlines()]

    def brief(r: dict | None) -> dict | None:
        return None if r is None else {k: v for k, v in r.items() if k not in ("per_crop_l1", "monitor")}

    summary = {"n": spec.n, "spec": asdict(spec), "meta": meta or {}, "init_sha256": state["init_sha256"],
               "final_sha256": state_dict_sha256(model), "steps_run": state["step"],
               "initial": brief(at[0]), "final": brief(at[spec.steps]),
               "best": {k: v for k, v in state["best"].items() if k != "fit_metrics"} | {"fit_metrics": state["best"].get("fit_metrics")},
               "matched_steps": {s: {"fit": brief(at.get(s)), "monitor": brief(mon_at.get(s))} for s in spec.milestones},
               "matched_exposure": {k: {"step": s, "fit": brief(at.get(s)), "monitor": brief(mon_at.get(s)), "lr": lr_at(s, spec.base_lr, spec.lr_drops)}
                                    for k, s in exp_steps.items()},
               "unreached_exposures": [k for k in spec.exposures if k not in exp_steps],
               "spikes": state["spike"]["events"] + ([state["spike"]["open"] | {"recovered_at_step": None}] if state["spike"]["open"] else []),
               "presentation_counts": list(sampler.counts), "passes": sampler.passes,
               "nonfinite_events": state["nonfinite_events"], "runtime_seconds": state["runtime_seconds"],
               "updates_per_second_median": float(np.median([r["updates_per_second"] for r in tl])) if tl else None,
               "train_loss_window_max_overall": max((r["train_loss_window_max"], r["step"]) for r in tl) if tl else None,
               "lineage": state["lineage"]}
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
