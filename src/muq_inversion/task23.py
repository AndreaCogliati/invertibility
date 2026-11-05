"""TASK-23: one-crop decoder-overfit diagnostic (helpers; the driver is scripts/task23_overfit.py).

One fixed training crop (first-conformer-block MuQ features -> BigVGAN-v2 log-mel) is fitted by
  A  a free tensor with the target's shape (bypasses the decoder: loss/mask/optimizer plumbing),
  B  a fresh TASK-22 MuQToMelTranslator with TASK-22's exact recipe at batch size one,
  C  B's initial weights with every explicit regularizer disabled (`disable_regularizers`).
Every update sees the identical cached pair. This tests fitting ability, not invertibility.

Facts verified in the TASK-22 code (scripts/train_task18_heldout.py, models/*):
- loss: `(pred - mel).abs().mean()` over all bins and frames; crops are exactly 187 mel hops of real
  audio, so no frame is padded and the valid mask is all ones (`masked_l1` equals it exactly then);
- no feature normalization (raw features into the input Linear) and no target normalization (raw
  log-mel); the only decode-time transform is the clamp to BigVGAN's log floor;
- MuQToMelTranslator has no dropout and no normalization layers (no buffers at all); Adam with
  weight_decay 0; no gradient clipping (`clip_grad_norm_(..., inf)` only measures); FP32.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

PAIR_TENSORS = ("features", "target_mel", "mask", "waveform", "positions")


class NonFiniteError(RuntimeError):
    """Raised (after diagnostics are saved) when a loss, gradient, or prediction is not finite."""


# --- cached pair -------------------------------------------------------------------------

def tensor_sha256(t: torch.Tensor) -> str:
    """SHA-256 over dtype, shape, and the raw bytes of a CPU-contiguous copy."""
    a = t.detach().cpu().contiguous()
    h = hashlib.sha256(f"{a.dtype}|{tuple(a.shape)}|".encode())
    h.update(a.numpy().tobytes())
    return h.hexdigest()


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


def check_pair_shapes(tensors: dict[str, torch.Tensor]) -> None:
    """features (1, T_muq, C) float32; target_mel (1, M, T) float32; mask (T,) bool, nonempty;
    waveform (N,) float32; positions (T,) float32 in [0, T_muq - 1]; all finite."""
    missing = set(PAIR_TENSORS) - set(tensors)
    if missing:
        raise ValueError(f"pair is missing {sorted(missing)}")
    f, y, m, w, p = (tensors[k] for k in PAIR_TENSORS)
    if f.ndim != 3 or f.shape[0] != 1 or y.ndim != 3 or y.shape[0] != 1:
        raise ValueError(f"features {tuple(f.shape)} / target_mel {tuple(y.shape)} must be (1, ., .)")
    t = y.shape[-1]
    if m.shape != (t,) or m.dtype != torch.bool or not bool(m.any()):
        raise ValueError(f"mask must be a nonempty bool ({t},), got {tuple(m.shape)} {m.dtype}")
    if p.shape != (t,) or float(p.min()) < 0 or float(p.max()) > f.shape[1] - 1:
        raise ValueError(f"positions must be ({t},) within [0, {f.shape[1] - 1}]")
    if w.ndim != 1:
        raise ValueError(f"waveform must be 1-D, got {tuple(w.shape)}")
    for k in ("features", "target_mel", "waveform", "positions"):
        if tensors[k].dtype != torch.float32 or not bool(torch.isfinite(tensors[k]).all()):
            raise ValueError(f"{k} must be finite float32")


def save_pair(path: Path, tensors: dict[str, torch.Tensor], meta: dict) -> dict[str, str]:
    """Atomically cache the pair with per-tensor hashes, reload, and verify. Returns the hashes."""
    check_pair_shapes(tensors)
    tensors = {k: tensors[k].detach().cpu().contiguous().clone() for k in PAIR_TENSORS}
    hashes = {k: tensor_sha256(v) for k, v in tensors.items()}
    atomic_torch_save(path, {"tensors": tensors, "meta": meta, "sha256": hashes})
    back = load_pair(path)
    for k in PAIR_TENSORS:
        if not torch.equal(back["tensors"][k], tensors[k]):
            raise RuntimeError(f"pair round trip failed for {k}")
    return hashes


def load_pair(path: Path) -> dict:
    """Load a cached pair; raises if any tensor no longer matches its recorded hash."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    for k in PAIR_TENSORS:
        got = tensor_sha256(d["tensors"][k])
        if got != d["sha256"][k]:
            raise RuntimeError(f"{path}: {k} hash {got[:12]} != recorded {d['sha256'][k][:12]}")
    check_pair_shapes(d["tensors"])
    return d


# --- metrics -----------------------------------------------------------------------------

def _frame_mask(mask: torch.Tensor, batch: int) -> torch.Tensor:
    m = mask if mask.ndim == 2 else mask[None].expand(batch, -1)
    if not bool(m.any()):
        raise ValueError("empty valid mask")
    return m


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/target (B, M, T); mask (T,) or (B, T) bool. Mean |pred - target| over valid frames and
    all bins. With an all-ones mask this is exactly TASK-22's `(pred - mel).abs().mean()`."""
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
    m = _frame_mask(mask, pred.shape[0]).to(pred.dtype)[:, None, :]
    return ((pred - target).abs() * m).sum() / (m.sum() * pred.shape[1])


def masked_diff_errors(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Diagnostic only (never a loss). Temporal: mean |dt pred - dt target| over adjacent frame pairs
    whose BOTH frames are valid; frequency: mean |df pred - df target| over adjacent bins of valid frames."""
    m = _frame_mask(mask, pred.shape[0])
    pair = (m[:, 1:] & m[:, :-1]).to(pred.dtype)[:, None, :]
    dt = ((pred[..., 1:] - pred[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs()
    df = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs()
    fm = m.to(pred.dtype)[:, None, :]
    return {"time_diff_l1": float((dt * pair).sum() / (pair.sum() * pred.shape[1])) if pair.sum() > 0 else float("nan"),
            "freq_diff_l1": float((df * fm).sum() / (fm.sum() * (pred.shape[1] - 1)))}


def boundary_frames(num_frames: int, edge: int, positions: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """(num_frames,) bool: frames within `edge` of either crop end (the translator's zero-padding
    reach, half its receptive field) or whose MuQ source position is edge-replicated (outside the
    span of token centers, `time_alignment.source_positions`)."""
    b = torch.zeros(num_frames, dtype=torch.bool)
    if edge > 0:
        b[:edge] = True
        b[-edge:] = True
    p = positions.detach().cpu()
    return b | (p <= 0) | (p >= num_tokens - 1)


def mel_diagnostics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, boundary: torch.Tensor) -> dict[str, float]:
    """Direct predicted-mel vs target-mel metrics BEFORE vocoding. pred/target (1, M, T)."""
    pred, target = pred.detach().float().cpu(), target.detach().float().cpu()
    mask, boundary = mask.cpu(), boundary.cpu()
    out = {"l1": float(masked_l1(pred, target, mask)), "nonfinite_pred": int((~torch.isfinite(pred)).sum()),
           "max_abs_err": float(((pred - target).abs() * mask[None, None]).max())}
    for name, sel in (("interior", mask & ~boundary), ("boundary", mask & boundary)):
        out[f"l1_{name}"] = float(masked_l1(pred, target, sel)) if bool(sel.any()) else float("nan")
    return out | masked_diff_errors(pred, target, mask)


# --- state-preserving train/eval comparison ----------------------------------------------

def _device_rng_module(device: torch.device):
    mod = getattr(torch, device.type, None) if device.type in ("cuda", "xpu", "mps") else None
    return mod if mod is not None and hasattr(mod, "get_rng_state") else None


@contextlib.contextmanager
def preserved_rng(device: torch.device) -> Iterator[None]:
    """Restore Python, NumPy, torch-CPU and the accelerator's RNG states on exit."""
    py, npst, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    mod = _device_rng_module(device)
    dev = mod.get_rng_state() if mod is not None else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(npst)
        torch.set_rng_state(cpu)
        if mod is not None:
            mod.set_rng_state(dev)


def mode_predictions(module: nn.Module, forward: Callable[[nn.Module], torch.Tensor],
                     device: torch.device) -> dict[str, torch.Tensor]:
    """forward(module) in training and in evaluation mode, under no_grad, without changing any
    persistent state: parameters/buffers (e.g. BatchNorm running statistics) are restored after
    each pass, RNG states are restored, and the module's mode is put back. Raises if anything changed."""
    was_training = module.training
    before = {k: v.detach().clone() for k, v in module.state_dict().items()}
    out = {}
    with preserved_rng(device), torch.no_grad():
        for mode in ("train", "eval"):
            module.train(mode == "train")
            out[mode] = forward(module).detach()
            module.load_state_dict(before)
    module.train(was_training)
    changed = [k for k, v in module.state_dict().items() if not torch.equal(v, before[k])]
    if changed:
        raise RuntimeError(f"diagnostic forward changed persistent state: {changed}")
    return out


# --- regularizers ------------------------------------------------------------------------

NORM_TYPES = (nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm)


def regularizer_audit(module: nn.Module, spec: ArmSpec) -> dict:
    """What in this model/optimizer could regularize or depend on the batch."""
    return {
        "dropout_modules": {n: m.p for n, m in module.named_modules() if isinstance(m, nn.modules.dropout._DropoutNd)},
        "normalization_modules": {n: type(m).__name__ for n, m in module.named_modules() if isinstance(m, NORM_TYPES)},
        "batch_dependent_normalization": [n for n, m in module.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)],
        "buffers": [n for n, _ in module.named_buffers()],
        "optimizer": "Adam", "weight_decay": spec.weight_decay,
    }


def disable_regularizers(module: nn.Module, spec: ArmSpec) -> list[str]:
    """Set every dropout probability and the optimizer weight decay to zero. Normalization layers
    and the architecture are never touched. Returns what was changed (empty if nothing was active)."""
    changed = []
    for n, m in module.named_modules():
        if isinstance(m, nn.modules.dropout._DropoutNd) and m.p > 0:
            changed.append(f"dropout {n}: p {m.p} -> 0")
            m.p = 0.0
    if spec.weight_decay:
        changed.append(f"Adam weight_decay {spec.weight_decay} -> 0")
        spec.weight_decay = 0.0
    return changed


class FreeMel(nn.Module):
    """Control A: a freely trainable (1, M, T) tensor, initialized at `mean + noise_std * N(0, 1)`
    with its own seeded generator (not at the target)."""

    def __init__(self, shape: tuple[int, int, int], mean: float, noise_std: float, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.mel = nn.Parameter(mean + noise_std * torch.randn(shape, generator=g))

    def forward(self) -> torch.Tensor:
        return self.mel


def state_dict_sha256(module: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(tensor_sha256(v).encode())
    return h.hexdigest()


# --- training ----------------------------------------------------------------------------

@dataclass
class ArmSpec:
    """One run. `steps` is the first budget; if `extension_steps` is set and the evaluation-mode L1
    at `steps` is still above `criterion_fraction` x the initial evaluation-mode L1, training
    continues to `extension_steps` with the LR multiplied by `extension_lr_factor` from then on.
    `stop_at_criterion` (control A) stops at the first report point that meets the criterion.
    `plateau` (control A only): {window_steps, min_rel_improvement, factor, max_reductions}."""

    name: str
    learning_rate: float
    steps: int
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    criterion_fraction: float = 0.01
    report_every: int = 100
    save_steps: tuple[int, ...] = (0, 100, 1000, 5000, 20000, 50000)
    latest_every_steps: int = 1000
    extension_steps: int | None = None
    extension_lr_factor: float = 0.1
    stop_at_criterion: bool = False
    plateau: dict | None = None
    notes: dict = field(default_factory=dict)


def _rng_states(device: torch.device) -> dict:
    mod = _device_rng_module(device)
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "device": mod.get_rng_state().cpu() if mod is not None else None}


def _set_rng_states(s: dict, device: torch.device) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch_cpu"])
    mod = _device_rng_module(device)
    if mod is not None and s.get("device") is not None:
        mod.set_rng_state(s["device"])


def train_arm(spec: ArmSpec, module: nn.Module, forward: Callable[[nn.Module], torch.Tensor],
              target: torch.Tensor, mask: torch.Tensor, boundary: torch.Tensor, run_dir: Path,
              device: torch.device, meta: dict | None = None, stop_after_step: int | None = None,
              should_stop: Callable[[], bool] = lambda: False, log: Callable[[str], None] = print) -> dict | None:
    """Fit forward(module) (1, M, T) to target (1, M, T) with masked L1 and Adam, resumably.

    run_dir/checkpoints: latest.pt (every `latest_every_steps` and on stop), milestone_XXXXXX.pt at
    `save_steps`, best.pt (lowest evaluation-mode L1 at a report point), final.pt. run_dir/preds:
    train- and eval-mode predictions at `save_steps` and best/final. run_dir/metrics.jsonl: one row
    per report point (every `report_every` updates). Returns the summary, or None if interrupted
    (latest.pt saved; calling again resumes)."""
    run_dir = Path(run_dir)
    ck_dir, pred_dir = run_dir / "checkpoints", run_dir / "preds"
    latest = ck_dir / "latest.pt"
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    target, mask_dev = target.to(device), mask.to(device)
    opt = torch.optim.Adam(module.parameters(), lr=spec.learning_rate, betas=tuple(spec.betas),
                           weight_decay=spec.weight_decay)
    t_start = time.perf_counter()

    def checkpoint(path: Path) -> None:
        atomic_torch_save(path, {"model_state_dict": module.state_dict(), "optimizer_state_dict": opt.state_dict(),
                                 "rng": _rng_states(device), "state": state, "spec": asdict(spec), "meta": meta or {}})
        back = torch.load(path, map_location="cpu", weights_only=False)
        assert back["state"]["step"] == state["step"], f"{path}: checkpoint validation failed"
        k0 = next(iter(back["model_state_dict"]))
        assert torch.equal(back["model_state_dict"][k0], module.state_dict()[k0].cpu()), f"{path}: tensor mismatch"

    def report(step: int, grad_norm: float | None, window: dict) -> dict:
        preds = mode_predictions(module, forward, device)
        diag = {mode: mel_diagnostics(p, target, mask, boundary) for mode, p in preds.items()}
        entry = {"step": step, "lr": state["lr"], "grad_norm": grad_norm,
                 "train_loss_window_mean": window["sum"] / window["n"] if window["n"] else None,
                 "train_loss_window_max": window["max"] if window["n"] else None,
                 "runtime_seconds": state["runtime_seconds"] + time.perf_counter() - t_start,
                 "nonfinite_events": state["nonfinite_events"],
                 **{f"{mode}/{k}": v for mode, d in diag.items() for k, v in d.items()},
                 "train_eval_max_abs_diff": float((preds["train"] - preds["eval"]).abs().max())}
        if any(d["nonfinite_pred"] for d in diag.values()):
            state["nonfinite_events"] += 1
            entry["nonfinite_events"] = state["nonfinite_events"]
        state["history"].append(entry)
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        if state["initial"] is None:
            state["initial"] = {"step": step, "eval_l1": entry["eval/l1"], "train_l1": entry["train/l1"]}
        if entry["eval/l1"] < state["best"]["eval_l1"]:
            state["best"] = {"step": step, "eval_l1": entry["eval/l1"]}
            atomic_torch_save(ck_dir / "best.pt", {"model_state_dict": module.state_dict(), "step": step,
                                                  "eval_l1": entry["eval/l1"], "spec": asdict(spec), "meta": meta or {}})
            atomic_torch_save(pred_dir / "best.pt", {k: v.cpu() for k, v in preds.items()} | {"step": step})
        if step in spec.save_steps:
            atomic_torch_save(pred_dir / f"step_{step:06d}.pt", {k: v.cpu() for k, v in preds.items()} | {"step": step})
        return entry

    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        module.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        _set_rng_states(ck["rng"], device)
        state = ck["state"]
        state["lineage"].append({"resumed_at_step": state["step"], "unix_time": time.time()})
        hist = state["history"]  # rows logged after latest.pt was written are dropped, then re-logged
        atomic_write_text(run_dir / "metrics.jsonl", "".join(json.dumps(e) + "\n" for e in hist))
        log(f"[{spec.name}] resumed from {latest} at step {state['step']}")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"{run_dir} exists without latest.pt; refusing to overwrite")
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {"step": 0, "lr": spec.learning_rate, "max_steps": spec.steps, "history": [], "initial": None,
                 "best": {"step": None, "eval_l1": float("inf")}, "criterion": None, "extension": None,
                 "plateau": {"reductions": [], "last_improvement_step": 0, "best_at_last_check": float("inf")},
                 "nonfinite_events": 0, "runtime_seconds": 0.0, "lineage": [{"started_at_step": 0, "unix_time": time.time()}],
                 "init_sha256": state_dict_sha256(module)}
        report(0, None, {"n": 0, "sum": 0.0, "max": 0.0})
        checkpoint(ck_dir / "milestone_000000.pt")
        checkpoint(latest)
    params = [p for p in module.parameters() if p.requires_grad]
    window = state.setdefault("window", {"n": 0, "sum": 0.0, "max": 0.0})  # checkpointed: resumed logs are identical
    while state["step"] < state["max_steps"]:
        step = state["step"] + 1
        module.train()
        pred = forward(module)
        loss = masked_l1(pred, target, mask_dev)
        if not torch.isfinite(loss):
            state["nonfinite_events"] += 1
            checkpoint(ck_dir / f"nonfinite_step_{step:06d}.pt")
            raise NonFiniteError(f"[{spec.name}] non-finite loss at step {step}; state saved")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float("inf")))  # measure only, no clipping
        if not np.isfinite(grad_norm):
            state["nonfinite_events"] += 1
            checkpoint(ck_dir / f"nonfinite_step_{step:06d}.pt")
            raise NonFiniteError(f"[{spec.name}] non-finite gradient norm at step {step}; state saved")
        for g in opt.param_groups:
            g["lr"] = state["lr"]
        opt.step()
        state["step"] = step
        lv = loss.item()
        window["n"], window["sum"], window["max"] = window["n"] + 1, window["sum"] + lv, max(window["max"], lv)
        if step % spec.report_every == 0 or step in spec.save_steps or step == state["max_steps"]:
            e = report(step, grad_norm, window)
            window.update(n=0, sum=0.0, max=0.0)
            thr = spec.criterion_fraction * state["initial"]["eval_l1"]
            if state["criterion"] is None and e["eval/l1"] <= thr:
                state["criterion"] = {"reached_at_step": step, "eval_l1": e["eval/l1"], "threshold": thr}
                log(f"[{spec.name}] criterion reached at step {step}: eval L1 {e['eval/l1']:.6f} <= {thr:.6f}")
            if spec.plateau and state["criterion"] is None:
                pl = state["plateau"]
                if e["eval/l1"] < pl["best_at_last_check"] * (1 - spec.plateau["min_rel_improvement"]):
                    pl["best_at_last_check"], pl["last_improvement_step"] = e["eval/l1"], step
                elif (step - pl["last_improvement_step"] >= spec.plateau["window_steps"]
                      and len(pl["reductions"]) < spec.plateau["max_reductions"]):
                    state["lr"] *= spec.plateau["factor"]
                    pl["reductions"].append({"step": step, "new_lr": state["lr"], "eval_l1": e["eval/l1"]})
                    pl["last_improvement_step"], pl["best_at_last_check"] = step, e["eval/l1"]
                    log(f"[{spec.name}] plateau at step {step}: LR -> {state['lr']:g}")
            if step % (spec.report_every * 10) == 0 or step in spec.save_steps:
                log(f"[{spec.name}] step {step}: eval L1 {e['eval/l1']:.6f} (train-mode {e['train/l1']:.6f}) | "
                    f"interior {e['eval/l1_interior']:.6f} boundary {e['eval/l1_boundary']:.6f} | grad {grad_norm:.3g} lr {state['lr']:g}")
        if step == spec.steps and spec.extension_steps and state["extension"] is None:
            last = state["history"][-1]
            assert last["step"] == step
            extend = last["eval/l1"] > spec.criterion_fraction * state["initial"]["eval_l1"]
            state["extension"] = {"decided_at_step": step, "eval_l1": last["eval/l1"],
                                  "threshold": spec.criterion_fraction * state["initial"]["eval_l1"], "extended": extend}
            if extend:
                state["max_steps"] = spec.extension_steps
                state["lr"] *= spec.extension_lr_factor
                log(f"[{spec.name}] above criterion at {step}: extending to {spec.extension_steps} with LR {state['lr']:g}")
        if step in spec.save_steps:
            checkpoint(ck_dir / f"milestone_{step:06d}.pt")
        if spec.stop_at_criterion and state["criterion"] is not None:
            break
        interrupted = should_stop() or step == stop_after_step
        if step % spec.latest_every_steps == 0 or interrupted:
            state["runtime_seconds"] += time.perf_counter() - t_start
            t_start = time.perf_counter()
            checkpoint(latest)
        if interrupted:
            log(f"[{spec.name}] interrupted at step {step}; latest.pt saved")
            return None
    if state["history"][-1]["step"] != state["step"]:
        report(state["step"], None, window)
    state["runtime_seconds"] += time.perf_counter() - t_start
    preds = mode_predictions(module, forward, device)
    atomic_torch_save(pred_dir / "final.pt", {k: v.cpu() for k, v in preds.items()} | {"step": state["step"]})
    checkpoint(ck_dir / "final.pt")
    checkpoint(latest)
    init, last = state["initial"], state["history"][-1]
    summary = {"arm": spec.name, "spec": asdict(spec), "meta": meta or {}, "init_sha256": state["init_sha256"],
               "final_sha256": state_dict_sha256(module), "steps_run": state["step"], "initial": init,
               "final": {k: last[k] for k in last if k.startswith(("eval/", "train/")) or k in ("step", "lr")},
               "best": state["best"], "best_relative_to_initial": state["best"]["eval_l1"] / init["eval_l1"],
               "final_relative_to_initial": last["eval/l1"] / init["eval_l1"], "criterion": state["criterion"],
               "criterion_met": state["criterion"] is not None, "extension": state["extension"],
               "plateau_reductions": state["plateau"]["reductions"], "nonfinite_events": state["nonfinite_events"],
               "max_train_eval_abs_diff": max(e["train_eval_max_abs_diff"] for e in state["history"]),
               "runtime_seconds": state["runtime_seconds"], "lineage": state["lineage"]}
    atomic_write_text(summary_path, json.dumps(summary, indent=2))
    return summary
