"""TASK-27: adversarial fine-tuning of the first-block MuQ -> mel translator (helpers; the driver is
scripts/task27_adversarial.py).

Two matched arms continue TASK-25 C's final generator on TASK-24's fixed N=256 pool:
  A  masked mel L1 only;
  B  masked mel L1 + lambda(u) * hinge adversarial loss from a conditional 2D PatchGAN discriminator.
Everything else (generator, start weights, sample stream, batch 16, LR schedule, exposure, evaluation
schedule) is identical. MuQ features are cached tensors (no gradient), BigVGAN is evaluation-only.

Discriminator input (identical preprocessing for real and generated mels; `d_input`):
  mel (B, M, T) -> clamp at BigVGAN's log-mel floor (what BigVGAN sees; the targets already satisfy it)
  -> fixed affine (x - center) / scale with constants from the training targets.
Conditioning: the cached MuQ features (B, T_muq, C) are moved to the mel frame grid with the SAME
deterministic frame-center linear interpolation the generator uses (`interpolate_at_positions` at
`positions`), then a spectrally normalized 1x1 Conv1d projects them to `cond_channels` per frame, which
are broadcast over the mel-bin axis and concatenated with the mel as extra 2D input planes.
All convolutions are unpadded ("valid"), so every logit's receptive field lies inside the crop. Logits
whose receptive field touches an invalid frame are masked (`logit_valid_mask`).
"""

from __future__ import annotations

import hashlib
import json
import math
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
from torch.nn.utils.parametrizations import spectral_norm

from muq_inversion.task23 import (
    NonFiniteError,
    _rng_states,
    _set_rng_states,
    atomic_torch_save,
    masked_l1,
    preserved_rng,
    state_dict_sha256,
)
from muq_inversion.task24 import ShuffledPassSampler, evaluate_set
from muq_inversion.task25 import lr_at_presentations, truncate_jsonl_presentations
from muq_inversion.time_alignment import interpolate_at_positions

# (kernel, stride) per conv layer, identical on the bin and frame axes; no padding.
D_LAYERS: tuple[tuple[int, int], ...] = ((3, 1), (4, 2), (4, 2), (3, 1), (3, 1))


# --- discriminator ------------------------------------------------------------------------

def receptive_field(layers: tuple[tuple[int, int], ...] = D_LAYERS) -> tuple[int, int]:
    """(receptive field, jump) in input cells for a stack of unpadded (kernel, stride) layers."""
    rf, jump = 1, 1
    for k, s in layers:
        rf += (k - 1) * jump
        jump *= s
    return rf, jump


def output_length(n: int, layers: tuple[tuple[int, int], ...] = D_LAYERS) -> int:
    for k, s in layers:
        n = (n - k) // s + 1
    return n


def align_conditioning(feats: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """feats (B, T_muq, C) -> (B, C, T_mel): the generator's frame-center linear interpolation, applied to the
    raw cached features (deterministic, no learned parameters, no gradient into the features)."""
    return interpolate_at_positions(feats.detach().transpose(1, 2), positions)


def d_input(mel: torch.Tensor, floor: float, center: float, scale: float) -> torch.Tensor:
    """The single preprocessing applied to every mel the discriminator sees (real or generated)."""
    return (mel.clamp(min=floor) - center) / scale


def logit_valid_mask(frame_mask: torch.Tensor, n_bins: int, layers: tuple[tuple[int, int], ...] = D_LAYERS) -> torch.Tensor:
    """frame_mask (T,) or (B, T) bool -> (B or 1, F', T') bool: True where the logit's receptive field lies
    entirely on valid frames (all mel bins are always valid)."""
    m = frame_mask if frame_mask.ndim == 2 else frame_mask[None]
    bad = (~m).float()[:, None, :]
    for k, s in layers:
        bad = F.max_pool1d(bad, k, s)
    t_valid = bad[:, 0] == 0
    return t_valid[:, None, :].expand(-1, output_length(n_bins, layers), -1)


class ConditionalPatchDiscriminator(nn.Module):
    """Conditional 2D PatchGAN: (B, M, T) mel + (B, C_in, T) aligned conditioning -> (B, F', T') logits.
    Spectral norm on every weight layer, LeakyReLU(0.2), no normalization layers, no sigmoid."""

    def __init__(self, cond_in: int, cond_channels: int = 16, channels: tuple[int, ...] = (32, 64, 128, 128), slope: float = 0.2):
        super().__init__()
        self.cond = spectral_norm(nn.Conv1d(cond_in, cond_channels, 1))
        widths = (1 + cond_channels, *channels, 1)
        self.convs = nn.ModuleList(spectral_norm(nn.Conv2d(widths[i], widths[i + 1], k, s))
                                   for i, (k, s) in enumerate(D_LAYERS))
        self.slope = slope

    def forward(self, mel: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if mel.shape[0] != cond.shape[0] or mel.shape[-1] != cond.shape[-1]:
            raise ValueError(f"mel {tuple(mel.shape)} and conditioning {tuple(cond.shape)} are not aligned")
        c = self.cond(cond)[:, :, None, :].expand(-1, -1, mel.shape[1], -1)
        x = torch.cat([mel[:, None], c], dim=1)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = F.leaky_relu(x, self.slope)
        return x[:, 0]

    def describe(self, n_mels: int, t_mel: int, hop_ms: float) -> dict:
        rf, jump = receptive_field()
        macs = self.cond.out_channels * self.cond.in_channels * t_mel
        f, t = n_mels, t_mel
        for conv in self.convs:
            (kf, kt), (sf, st) = conv.kernel_size, conv.stride
            f, t = (f - kf) // sf + 1, (t - kt) // st + 1
            macs += conv.out_channels * conv.in_channels * kf * kt * f * t
        return {"layers": [f"cond: SN Conv1d({self.cond.in_channels}->{self.cond.out_channels}, k=1), broadcast over {n_mels} bins"]
                + [f"SN Conv2d({c.in_channels}->{c.out_channels}, k={c.kernel_size[0]}, s={c.stride[0]}, pad 0)" + (", LeakyReLU 0.2" if i < len(self.convs) - 1 else ", linear logits")
                   for i, c in enumerate(self.convs)],
                "parameters": sum(p.numel() for p in self.parameters()), "logit_grid": [output_length(n_mels), output_length(t_mel)],
                "receptive_field_bins": rf, "receptive_field_frames": rf, "receptive_field_ms": rf * hop_ms, "logit_stride": jump,
                "fits_crop": rf <= t_mel and rf <= n_mels, "forward_macs_per_crop": macs}


def build_discriminator(cfg_d: dict, cond_in: int, seed: int) -> ConditionalPatchDiscriminator:
    """Deterministic initialization from its own seed, without touching the global RNG streams."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ConditionalPatchDiscriminator(cond_in, cfg_d["cond_channels"], tuple(cfg_d["channels"]), cfg_d["leaky_slope"])


def verify_start_checkpoint(model: nn.Module, ck: dict, expected: dict) -> str:
    """The common start must be exactly the recorded checkpoint: state-dict SHA-256, its training spec fields and
    update count. Raises ValueError (never substitutes). Returns the SHA-256."""
    sha = state_dict_sha256(model)
    if sha != expected["expected_sha256"]:
        raise ValueError(f"start checkpoint state-dict SHA-256 {sha} != expected {expected['expected_sha256']}")
    spec = ck.get("spec", {})
    for k, v in expected["expected_spec"].items():
        got = spec.get(k)
        if (list(got) if isinstance(got, tuple) else got) != v:
            raise ValueError(f"start checkpoint spec {k}={got!r}, expected {v!r}")
    if ck.get("state", {}).get("update") != expected["expected_update"]:
        raise ValueError(f"start checkpoint is at update {ck.get('state', {}).get('update')}, expected {expected['expected_update']}")
    return sha


# --- losses and schedule ------------------------------------------------------------------

def masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    v = valid.expand_as(x).to(x.dtype)
    return (x * v).sum() / v.sum()


def d_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """mean(relu(1 - D(z, m))) and mean(relu(1 + D(z, m_hat))) over valid logits (m_hat must be detached)."""
    return masked_mean(F.relu(1 - real_logits), valid), masked_mean(F.relu(1 + fake_logits), valid)


def g_hinge_loss(fake_logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return -masked_mean(fake_logits, valid)


def lambda_at(update: int, lam: float, ramp_updates: int) -> float:
    """Weight for generator update `update` (1-based): 0 at the first update, linear to `lam` over
    `ramp_updates` updates, then fixed."""
    if ramp_updates <= 0:
        return lam
    return lam * min(1.0, (update - 1) / ramp_updates)


def _grad_list(loss: torch.Tensor, params: list[torch.Tensor], retain: bool) -> list[torch.Tensor]:
    return [g if g is not None else torch.zeros_like(p) for g, p in
            zip(torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True), params, strict=True)]


def _norm(gs: list[torch.Tensor]) -> float:
    return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs)))


class FrozenD:
    """Context: discriminator in eval mode (no spectral-norm power iteration) with requires_grad off, so a
    generator step neither updates nor accumulates into it; gradients still flow to its input."""

    def __init__(self, d: nn.Module):
        self.d = d

    def __enter__(self):
        self.was = self.d.training
        self.flags = [p.requires_grad for p in self.d.parameters()]
        self.d.eval()
        for p in self.d.parameters():
            p.requires_grad_(False)
        return self.d

    def __exit__(self, *exc):
        for p, f in zip(self.d.parameters(), self.flags, strict=True):
            p.requires_grad_(f)
        self.d.train(self.was)


def d_step(d: nn.Module, opt_d: torch.optim.Optimizer, real: torch.Tensor, fake: torch.Tensor, cond: torch.Tensor,
           valid: torch.Tensor, prep: Callable[[torch.Tensor], torch.Tensor], lr: float) -> dict:
    """One discriminator update. Real and (detached) generated mels share one batch and identical conditioning."""
    d.train()
    n = real.shape[0]
    logits = d(prep(torch.cat([real, fake.detach()])), torch.cat([cond, cond]))
    lr_, lf = d_hinge_loss(logits[:n], logits[n:], valid)
    loss = lr_ + lf
    if not torch.isfinite(loss):
        raise NonFiniteError("non-finite discriminator loss")
    opt_d.zero_grad(set_to_none=True)
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(list(d.parameters()), float("inf")))  # measure only
    if not math.isfinite(gn):
        raise NonFiniteError("non-finite discriminator gradient norm")
    for g in opt_d.param_groups:
        g["lr"] = lr
    opt_d.step()
    with torch.no_grad():
        vr, vf = valid.expand_as(logits[:n]), valid.expand_as(logits[n:])
        real_l, fake_l = logits[:n][vr], logits[n:][vf]
        return {"d_loss": float(loss), "d_loss_real": float(lr_), "d_loss_fake": float(lf), "d_grad_norm": gn,
                "d_real_logit_mean": float(real_l.mean()), "d_fake_logit_mean": float(fake_l.mean()),
                "d_real_saturated_frac": float((real_l >= 1).float().mean()), "d_fake_saturated_frac": float((fake_l <= -1).float().mean()),
                "d_accuracy": float(((real_l > 0).float().mean() + (fake_l < 0).float().mean()) / 2)}


def g_adv_loss(d: nn.Module, pred: torch.Tensor, cond: torch.Tensor, valid: torch.Tensor,
               prep: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    with FrozenD(d) as fd:
        return g_hinge_loss(fd(prep(pred), cond), valid)


# --- lambda calibration --------------------------------------------------------------------

def calibrate_lambda(g: nn.Module, d: nn.Module, opt_d: torch.optim.Optimizer, data: dict, cfg_cal: dict, d_lr: float,
                     batch: int, prep: Callable, valid: torch.Tensor, device: torch.device) -> dict:
    """Temporary generator `g` (start weights, not updated) and discriminator `d`: `warmup` D-only updates,
    then on `n_batches` fixed training batches the generator gradient norms of the L1 and adversarial terms.
    lambda = target_ratio * median(||g_L1||) / median(||g_adv||). Stops (raises) if degenerate."""
    feats, mel, pos, mask = data["fit_feats"], data["fit_mel"], data["positions"], data["mask"].to(device)
    t_mel = mel.shape[-1]
    params = [p for p in g.parameters() if p.requires_grad]
    before = state_dict_sha256(g)
    warm = ShuffledPassSampler(feats.shape[0], cfg_cal["warmup_sampler_seed"])
    warm_log = []
    for _ in range(cfg_cal["warmup_updates"]):
        idx = torch.tensor([warm.next() for _ in range(batch)], device=feats.device)
        with torch.no_grad():
            g.train()
            fake = g(feats[idx], t_mel, pos)
        warm_log.append(d_step(d, opt_d, mel[idx], fake, align_conditioning(feats[idx], pos), valid, prep, d_lr))
    cal = ShuffledPassSampler(feats.shape[0], cfg_cal["batch_sampler_seed"])
    rows = []
    for _ in range(cfg_cal["n_batches"]):
        idx = torch.tensor([cal.next() for _ in range(batch)], device=feats.device)
        g.train()
        pred = g(feats[idx], t_mel, pos)
        l1 = masked_l1(pred, mel[idx], mask)
        adv = g_adv_loss(d, pred, align_conditioning(feats[idx], pos), valid, prep)
        n1, n2 = _norm(_grad_list(l1, params, True)), _norm(_grad_list(adv, params, False))
        rows.append({"indices": idx.tolist(), "l1": l1.item(), "adv": adv.item(), "grad_norm_l1": n1, "grad_norm_adv": n2,
                     "ratio_adv_over_l1": n2 / n1 if n1 > 0 else float("nan")})
    if state_dict_sha256(g) != before:
        raise RuntimeError("calibration changed the generator")
    g1, g2 = np.array([r["grad_norm_l1"] for r in rows]), np.array([r["grad_norm_adv"] for r in rows])
    out = {"warmup": {"updates": len(warm_log), "first": warm_log[0] if warm_log else None, "last": warm_log[-1] if warm_log else None},
           "batches": rows, "median_grad_norm_l1": float(np.median(g1)), "median_grad_norm_adv": float(np.median(g2)),
           "target_ratio": cfg_cal["target_ratio"]}
    problems = []
    if not (np.isfinite(g1).all() and np.isfinite(g2).all()):
        problems.append("non-finite gradient norms")
    if np.median(g1) <= cfg_cal["min_grad_norm"] or np.median(g2) <= cfg_cal["min_grad_norm"]:
        problems.append(f"degenerate gradient norm (median L1 {np.median(g1):.3g}, adv {np.median(g2):.3g})")
    last = out["warmup"]["last"] or {}
    if last.get("d_real_saturated_frac", 0) > cfg_cal["max_saturation"] and last.get("d_fake_saturated_frac", 0) > cfg_cal["max_saturation"]:
        problems.append(f"discriminator saturated after warm-up ({last})")
    out["problems"] = problems
    if problems:
        out["lambda"] = None
        return out
    lam = cfg_cal["target_ratio"] * float(np.median(g1)) / float(np.median(g2))
    out["lambda"] = lam
    out["weighted_adv_over_l1_median"] = float(np.median(lam * g2 / g1))
    return out


# --- training -----------------------------------------------------------------------------

@dataclass
class AdvSpec:
    name: str
    adversarial: bool
    batch: int
    base_lr: float
    betas: tuple[float, float]
    total_presentations: int
    lr_drops: tuple[tuple[int, float], ...]
    lam: float = 0.0
    ramp_fraction: float = 0.1
    d_lr: float = 1e-4
    d_betas: tuple[float, float] = (0.5, 0.9)
    d_warmup_updates: int = 100
    d_warmup_sampler_seed: str = "task27-dwarmup-0"
    d_init_seed: int = 0
    log_every: int = 1000
    fit_eval_every: int = 16384
    monitor_eval_every: int = 65536
    latest_every: int = 65536
    chunk: int = 64
    sampler_seed: str = "task27-sampler-0"
    spike_factor: float = 2.0
    recovery_factor: float = 1.1
    anchor_index: int = 0
    showcase: tuple[int, ...] = (0, 1, 2, 3)

    @property
    def updates(self) -> int:
        if self.total_presentations % self.batch:
            raise ValueError("the budget must be a whole number of batches")
        return self.total_presentations // self.batch

    @property
    def ramp_updates(self) -> int:
        return round(self.ramp_fraction * self.updates)

    def boundary_updates(self) -> dict[int, int]:
        return {at: at // self.batch for at, _ in self.lr_drops if at % self.batch == 0}

    def special_updates(self) -> set[int]:
        return set(self.boundary_updates().values()) | {self.ramp_updates, self.updates}


def shuffled_features(feats: torch.Tensor) -> torch.Tensor:
    """Conditioning control: every crop gets the next crop's features (a derangement for n >= 2)."""
    return torch.roll(feats, shifts=1, dims=0)


@torch.no_grad()
def d_scores(d: nn.Module, feats: torch.Tensor, real: torch.Tensor, fake: torch.Tensor, positions: torch.Tensor,
             valid: torch.Tensor, prep: Callable, device: torch.device, chunk: int) -> dict:
    """Mean discriminator logits on real and generated mels, eval mode, no state change."""
    before = {k: v.detach().clone() for k, v in d.state_dict().items()}
    was = d.training
    d.eval()
    r, f = [], []
    with preserved_rng(device):
        for i in range(0, feats.shape[0], chunk):
            c = align_conditioning(feats[i:i + chunk].to(device), positions)
            v = valid
            r.append(masked_mean(d(prep(real[i:i + chunk].to(device)), c), v).item() * min(chunk, feats.shape[0] - i))
            f.append(masked_mean(d(prep(fake[i:i + chunk].to(device)), c), v).item() * min(chunk, feats.shape[0] - i))
    d.train(was)
    if any(not torch.equal(v, before[k]) for k, v in d.state_dict().items()):
        raise RuntimeError("discriminator scoring changed its state")
    n = feats.shape[0]
    return {"d_real_logit_mean": sum(r) / n, "d_fake_logit_mean": sum(f) / n}


def train_adversarial_arm(spec: AdvSpec, g: nn.Module, d: nn.Module | None, data: dict, run_dir: Path, device: torch.device,
                          prep: Callable, logit_valid: torch.Tensor, meta: dict | None = None, stop_after_update: int | None = None,
                          should_stop: Callable[[], bool] = lambda: False, log: Callable[[str], None] = print,
                          metrics_sink: Callable[[str, dict, int], None] | None = None,
                          state_hook: Callable[[dict], None] | None = None) -> dict | None:
    """Resumable continuation of `g` for exactly spec.total_presentations; arm B also trains `d` (after
    `d_warmup_updates` D-only updates). data: fit_feats, fit_mel, mon_feats, mon_mel, mask, boundary,
    positions. Writes train_log / fit_eval / monitor_eval JSONL, checkpoints (init, latest, best, final,
    lr_boundary_*), preds (start, best, final), summary.json."""
    run_dir = Path(run_dir)
    ck, pr = run_dir / "checkpoints", run_dir / "preds"
    latest, summary_path = ck / "latest.pt", run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    if spec.adversarial != (d is not None):
        raise ValueError("arm definition and discriminator presence disagree")
    fit_feats, fit_mel, mon_feats, mon_mel = (data[k] for k in ("fit_feats", "fit_mel", "mon_feats", "mon_mel"))
    mask, boundary, positions = data["mask"], data["boundary"], data["positions"]
    n, t_mel = fit_feats.shape[0], fit_mel.shape[-1]
    mask_dev = mask.to(device)
    valid = logit_valid.to(device)
    params = [p for p in g.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=spec.base_lr, betas=tuple(spec.betas), weight_decay=0.0)
    opt_d = torch.optim.Adam(d.parameters(), lr=spec.d_lr, betas=tuple(spec.d_betas), weight_decay=0.0) if d is not None else None
    sampler = ShuffledPassSampler(n, spec.sampler_seed)
    specials, boundaries = spec.special_updates(), spec.boundary_updates()
    files = {k: run_dir / f"{k}.jsonl" for k in ("train_log", "fit_eval", "monitor_eval")}

    def write(kind: str, row: dict) -> None:
        with open(files[kind], "a") as f:
            f.write(json.dumps(row) + "\n")
        if metrics_sink is not None:
            metrics_sink(kind, row, row["presentations"])

    def checkpoint(path: Path) -> None:
        obj = {"model_state_dict": g.state_dict(), "optimizer_state_dict": opt.state_dict(), "sampler_state": sampler.state_dict(),
               "rng": _rng_states(device), "state": state, "spec": asdict(spec), "meta": meta or {}}
        if d is not None:
            obj |= {"discriminator_state_dict": d.state_dict(), "discriminator_optimizer_state_dict": opt_d.state_dict()}
        atomic_torch_save(path, obj)
        back = torch.load(path, map_location="cpu", weights_only=False)
        assert back["state"]["update"] == state["update"] and back["sampler_state"]["counts"] == sampler.counts, f"{path}: validation failed"
        k0 = next(iter(back["model_state_dict"]))
        assert torch.equal(back["model_state_dict"][k0], g.state_dict()[k0].cpu()), f"{path}: tensor mismatch"

    def wall() -> float:
        return state["runtime_seconds"] + time.perf_counter() - t0

    def evaluate(update: int, monitor: bool) -> tuple[dict, torch.Tensor, torch.Tensor | None]:
        p = update * spec.batch
        fm, per_crop, fp = evaluate_set(g, fit_feats, fit_mel, mask, boundary, positions, device, spec.chunk)
        row = {"presentations": p, "update": update, "lr": state["lr"], "lambda": state["lambda"], "wall_seconds": wall(),
               **fm, "anchor_l1": per_crop[spec.anchor_index], "per_crop_l1": per_crop}
        mp = None
        if monitor:
            sf, _, _ = evaluate_set(g, shuffled_features(fit_feats), fit_mel, mask, boundary, positions, device, spec.chunk)
            row["shuffled_conditioning_l1"] = sf["l1"]
            mm, mpc, mp = evaluate_set(g, mon_feats, mon_mel, mask, boundary, positions, device, spec.chunk)
            ms, _, _ = evaluate_set(g, shuffled_features(mon_feats), mon_mel, mask, boundary, positions, device, spec.chunk)
            mrow = {"presentations": p, "update": update, "lr": state["lr"], "wall_seconds": wall(), **mm,
                    "shuffled_conditioning_l1": ms["l1"], "per_crop_l1": mpc}
            if d is not None:
                row |= {f"eval_{k}": v for k, v in d_scores(d, fit_feats, fit_mel, fp, positions, valid, prep, device, spec.chunk).items()}
                mrow |= {f"eval_{k}": v for k, v in d_scores(d, mon_feats, mon_mel, mp, positions, valid, prep, device, spec.chunk).items()}
            write("monitor_eval", mrow)
            row["monitor"] = {k: v for k, v in mrow.items() if k != "per_crop_l1"}
        write("fit_eval", row)
        sp = state["spike"]
        if sp["open"] is None and update > 0 and fm["l1"] > spec.spike_factor * state["best"]["fit_l1"]:
            sp["open"] = {"presentations": p, "update": update, "fit_l1": fm["l1"], "pre_spike_best": state["best"]["fit_l1"]}
            log(f"[{spec.name}] spike at {p:,} presentations: fit L1 {fm['l1']:.4f} vs best {state['best']['fit_l1']:.4f}")
        elif sp["open"] is not None:
            sp["open"]["peak_fit_l1"] = max(sp["open"].get("peak_fit_l1", sp["open"]["fit_l1"]), fm["l1"])
            if fm["l1"] <= spec.recovery_factor * sp["open"]["pre_spike_best"]:
                sp["events"].append({**sp["open"], "recovered_at_presentations": p, "recovery_presentations": p - sp["open"]["presentations"]})
                sp["open"] = None
        if update > 0 and fm["l1"] < state["best"]["fit_l1"]:
            state["best"] = {"presentations": p, "update": update, "fit_l1": fm["l1"], "anchor_l1": per_crop[spec.anchor_index], "per_crop_l1": per_crop}
            atomic_torch_save(ck / "best.pt", {"model_state_dict": g.state_dict(), "update": update, "presentations": p, "fit_l1": fm["l1"],
                                               "spec": asdict(spec), "meta": meta or {}})
            _, _, bmp = evaluate_set(g, mon_feats, mon_mel, mask, boundary, positions, device, spec.chunk) if mp is None else (None, None, mp)
            atomic_torch_save(pr / "best.pt", {"update": update, "presentations": p, "fit": fp, "monitor": bmp})
        elif update == 0:
            state["best"]["start_fit_l1"] = fm["l1"]
        return row, fp, mp

    if latest.exists():
        c = torch.load(latest, map_location="cpu", weights_only=False)
        g.load_state_dict(c["model_state_dict"])
        opt.load_state_dict(c["optimizer_state_dict"])
        if d is not None:
            d.load_state_dict(c["discriminator_state_dict"])
            opt_d.load_state_dict(c["discriminator_optimizer_state_dict"])
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
        log(f"[{spec.name}] resumed at update {state['update']:,}")
    else:
        if run_dir.exists() and any(p.name != "LOCK" for p in run_dir.iterdir()):
            raise SystemExit(f"{run_dir} has files but no latest.pt; refusing to overwrite")
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {"update": 0, "lr": spec.base_lr, "d_lr": spec.d_lr if d is not None else None, "lambda": 0.0,
                 "best": {"update": None, "fit_l1": float("inf")}, "spike": {"open": None, "events": []},
                 "runtime_seconds": 0.0, "nonfinite_events": 0, "order_sha256": hashlib.sha256().hexdigest(),
                 "window": {"n": 0}, "start_sha256": state_dict_sha256(g),
                 "lineage": [{"started_at_update": 0, "utc": datetime.now(UTC).isoformat(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}]}
        t0 = time.perf_counter()
        if state_hook:
            state_hook(state)
        if d is not None:  # D-only warm-up on training pairs, generator fixed (its own sample stream; no generator exposure)
            warm = ShuffledPassSampler(n, spec.d_warmup_sampler_seed)
            wl = []
            for _ in range(spec.d_warmup_updates):
                idx = torch.tensor([warm.next() for _ in range(spec.batch)], device=fit_feats.device)
                with torch.no_grad():
                    g.train()
                    fake = g(fit_feats[idx], t_mel, positions)
                wl.append(d_step(d, opt_d, fit_mel[idx], fake, align_conditioning(fit_feats[idx], positions), valid, prep, spec.d_lr))
            state["d_warmup"] = {"updates": len(wl), "first": wl[0], "last": wl[-1], "d_sha256_after": state_dict_sha256(d)}
            log(f"[{spec.name}] D warm-up: {len(wl)} updates, loss {wl[0]['d_loss']:.3f} -> {wl[-1]['d_loss']:.3f}, "
                f"accuracy {wl[-1]['d_accuracy']:.3f}")
            if state_dict_sha256(g) != state["start_sha256"]:
                raise RuntimeError("D warm-up changed the generator")
        atomic_torch_save(ck / "init.pt", {"model_state_dict": g.state_dict(), "sha256": state["start_sha256"], "spec": asdict(spec)}
                          | ({"discriminator_state_dict": d.state_dict()} if d is not None else {}))
        _, fp, mp = evaluate(0, monitor=True)
        atomic_torch_save(pr / "start.pt", {"update": 0, "fit": fp, "monitor": mp})
        checkpoint(latest)

    def new_window() -> dict:
        return {"n": 0, "l1_sum": 0.0, "l1_max": 0.0, "g_l1_grad_max": 0.0, "g_adv_grad_max": 0.0, "g_grad_max": 0.0, "adv_sum": 0.0,
                "d": {}}

    w = state["window"] if state["window"].get("n") else new_window()
    t_log, u_log = time.perf_counter(), state["update"]
    while state["update"] < spec.updates:
        u = state["update"] + 1
        before = (u - 1) * spec.batch
        idx_list = [sampler.next() for _ in range(spec.batch)]
        state["order_sha256"] = hashlib.sha256((state["order_sha256"] + json.dumps(idx_list)).encode()).hexdigest()
        idx = torch.tensor(idx_list, device=fit_feats.device)
        lr = lr_at_presentations(before, spec.base_lr, spec.lr_drops)
        lam = lambda_at(u, spec.lam, spec.ramp_updates) if d is not None else 0.0
        g.train()
        pred = g(fit_feats[idx], t_mel, positions)
        l1 = masked_l1(pred, fit_mel[idx], mask_dev)
        drow = {}
        if not torch.isfinite(l1):
            state["nonfinite_events"] += 1
            checkpoint(ck / f"nonfinite_{u:07d}.pt")
            raise NonFiniteError(f"[{spec.name}] non-finite L1 at update {u}; state saved")
        if d is not None:
            cond = align_conditioning(fit_feats[idx], positions)
            d_lr = lr_at_presentations(before, spec.d_lr, spec.lr_drops)
            try:
                drow = d_step(d, opt_d, fit_mel[idx], pred, cond, valid, prep, d_lr)
            except NonFiniteError as exc:
                state["nonfinite_events"] += 1
                checkpoint(ck / f"nonfinite_{u:07d}.pt")
                raise NonFiniteError(f"[{spec.name}] {exc} at update {u}; state saved") from exc
            state["d_lr"] = d_lr
            adv = g_adv_loss(d, pred, cond, valid, prep)
            g1 = _grad_list(l1, params, True)
            g2 = _grad_list(adv, params, False)
            grads = [a + lam * b for a, b in zip(g1, g2, strict=True)]
            n1, n2, nt, av = _norm(g1), _norm(g2), _norm(grads), adv.item()
        else:
            g1 = _grad_list(l1, params, False)
            grads, n1, n2, nt, av = g1, _norm(g1), 0.0, None, None
            nt = n1
        if not (math.isfinite(nt) and (av is None or math.isfinite(av))):
            state["nonfinite_events"] += 1
            checkpoint(ck / f"nonfinite_{u:07d}.pt")
            raise NonFiniteError(f"[{spec.name}] non-finite generator gradient/adversarial loss at update {u}; state saved")
        for p_, gr in zip(params, grads, strict=True):
            p_.grad = gr
        for grp in opt.param_groups:
            grp["lr"] = lr
        opt.step()
        opt.zero_grad(set_to_none=True)
        state["update"], state["lr"], state["lambda"] = u, lr, lam
        p = u * spec.batch
        lv = l1.item()
        w["n"] += 1
        w["l1_sum"] += lv
        w["l1_max"] = max(w["l1_max"], lv)
        w["g_l1_grad_max"] = max(w["g_l1_grad_max"], n1)
        w["g_adv_grad_max"] = max(w["g_adv_grad_max"], n2)
        w["g_grad_max"] = max(w["g_grad_max"], nt)
        if av is not None:
            w["adv_sum"] += av
        for k, v in drow.items():
            acc = w["d"].setdefault(k, {"sum": 0.0, "max": -math.inf, "min": math.inf})
            acc["sum"] += v
            acc["max"], acc["min"] = max(acc["max"], v), min(acc["min"], v)
        state["window"] = w
        if p // spec.log_every > before // spec.log_every:
            now = time.perf_counter()
            cts = np.array(sampler.counts)
            ups = (u - u_log) / max(now - t_log, 1e-9)
            row = {"presentations": p, "update": u, "lr": lr, "lambda": lam, "train_l1_window_mean": w["l1_sum"] / w["n"],
                   "train_l1_window_max": w["l1_max"], "g_grad_norm_l1": n1, "g_grad_norm_adv": n2, "g_grad_norm_total": nt,
                   "g_grad_norm_l1_window_max": w["g_l1_grad_max"], "g_grad_norm_adv_window_max": w["g_adv_grad_max"],
                   "g_grad_norm_total_window_max": w["g_grad_max"], "updates_per_second": ups, "presentations_per_second": ups * spec.batch,
                   "wall_seconds": wall(), "presentations_min": int(cts.min()), "presentations_max": int(cts.max())}
            if d is not None:
                row |= {"g_adv_window_mean": w["adv_sum"] / w["n"], "weighted_adv_over_l1_grad": lam * n2 / n1 if n1 > 0 else None,
                        "d_lr": state["d_lr"]}
                row |= {f"{k}_window_mean": a["sum"] / w["n"] for k, a in w["d"].items()}
                row |= {"d_grad_norm_window_max": w["d"]["d_grad_norm"]["max"],
                        "d_real_saturated_frac_window_max": w["d"]["d_real_saturated_frac"]["max"],
                        "d_fake_saturated_frac_window_max": w["d"]["d_fake_saturated_frac"]["max"]}
            write("train_log", row)
            w = new_window()
            state["window"] = w
            t_log, u_log = now, u
        if p % spec.fit_eval_every == 0 or u in specials:
            row, fp, mp = evaluate(u, monitor=p % spec.monitor_eval_every == 0 or u in specials)
            if p % (spec.fit_eval_every * 10) == 0 or u in specials:
                log(f"[{spec.name}] {p:,} presentations (update {u:,}): fit L1 {row['l1']:.5f} anchor {row['anchor_l1']:.5f}"
                    + (f" | monitor {row['monitor']['l1']:.4f}" if "monitor" in row else "") + f" | lr {lr:g} | lambda {lam:.4g}")
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
    summary = {"arm": spec.name, "spec": asdict(spec), "meta": meta or {}, "start_sha256": state["start_sha256"],
               "final_sha256": state_dict_sha256(g), "discriminator_final_sha256": state_dict_sha256(d) if d is not None else None,
               "d_warmup": state.get("d_warmup"), "order_sha256": state["order_sha256"],
               "updates_run": state["update"], "presentations": state["update"] * spec.batch,
               "start": brief(at_u[0]), "start_monitor": brief(mon_u.get(0)), "final": brief(at_u[spec.updates]),
               "final_monitor": brief(mon_u.get(spec.updates)), "best": state["best"],
               "lr_boundaries": {at: {"update": bu, "fit": brief(at_u.get(bu)), "monitor": brief(mon_u.get(bu))} for at, bu in boundaries.items()},
               "spikes": state["spike"]["events"] + ([state["spike"]["open"] | {"recovered_at_presentations": None}] if state["spike"]["open"] else []),
               "presentation_counts_min_max": [min(sampler.counts), max(sampler.counts)], "passes": sampler.passes,
               "nonfinite_events": state["nonfinite_events"], "runtime_seconds": state["runtime_seconds"],
               "updates_per_second_median": float(np.median([r["updates_per_second"] for r in tl])) if tl else None,
               "g_grad_norm_window_max_overall": max(r["g_grad_norm_total_window_max"] for r in tl) if tl else None,
               "lineage": state["lineage"], "wandb_run_id": state.get("wandb_run_id")}
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
