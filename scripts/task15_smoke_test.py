"""Bounded backend smoke test of the TASK-15/16 MuQ -> BigVGAN pipeline on one device.

Exercises every stage once, on the N=2 crops, with a handful of optimizer steps (no real training):

    1. MuQ forward                       6. BigVGAN inference (oracle + predicted)
    2. corrected MuQ->mel time alignment 7. optimizer step (loss must stay finite)
    3. translator forward/backward       8. checkpoint save -> portable load (CPU and device)
    4. raw predicted mel                 9. shuffled-across control
    5. decode-time mel-floor clamp

    uv run python scripts/task15_smoke_test.py --device mps
    uv run python scripts/task15_smoke_test.py --device cuda
    uv run python scripts/task15_smoke_test.py --device xpu
    uv run python scripts/task15_smoke_test.py --device cpu

Uses the TASK-16 config (corrected alignment + decode clamp) unless --config says otherwise; only
`device_request` is overridable. Writes `report.json` and the checkpoint to a fresh timestamped
directory under artifacts/backend_smoke/ (never reused). Exits 1 if any stage fails, with the
stage name and device attached to the error. Not a scientific result: numbers after a few steps
only show that each stage runs and produces finite, correctly shaped tensors.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

from muq_inversion.bigvgan_common import (
    bigvgan_log_mel_floor,
    clamp_to_mel_floor,
    load_bigvgan,
    load_mel_crops,
    mel_l1,
    shuffled_across_pairing,
)
from muq_inversion.device import (
    DEVICE_CHOICES,
    backend_stage,
    device_metadata,
    load_portable,
    memory_stats,
    require_fp32_default,
    seed_everything,
    synchronize,
)
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.logging_utils import capture_environment_manifest, select_device
from muq_inversion.models.muq_mel_translator import MuQToMelTranslator
from muq_inversion.time_alignment import (
    bigvgan_mel_frame_centers,
    muq_token_centers,
    source_positions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/task16_bigvgan_corrected.yaml"


def finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"{name}: non-finite values")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=DEVICE_CHOICES, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--n", default="2", help="manifest key in the config (default 2)")
    parser.add_argument("--steps", type=int, default=3, help="optimizer steps (bounded; default 3)")
    args = parser.parse_args()
    if not 1 <= args.steps <= 50:
        raise SystemExit("--steps must be in [1, 50]; this is a smoke test, not training")
    cfg = yaml.safe_load(args.config.read_text())

    device = select_device(args.device, cfg["allow_cpu_fallback"])
    require_fp32_default()
    seeding = seed_everything(cfg["seed"], device)
    info = device_metadata(args.device, device)
    print(f"device: {json.dumps(info)}")
    out = REPO_ROOT / "artifacts" / "backend_smoke" / f"{device.type}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=False)

    stages: dict[str, dict] = {}
    report = {"device_metadata": info, "seeding": seeding, "config": str(args.config.relative_to(REPO_ROOT)),
              "n": args.n, "steps": args.steps, "stages": stages,
              "environment": capture_environment_manifest(REPO_ROOT)}

    def stage(name: str):
        print(f"-- {name}")
        stages[name] = {"status": "running"}
        return backend_stage(name, device)

    t_start = time.perf_counter()
    try:
        with stage("load_models"):
            bigvgan = load_bigvgan(device)
            muq = FrozenMuQEncoder(cfg["muq_model_id"], device)
            h = bigvgan.h
            assert not any(p.requires_grad for p in bigvgan.parameters()), "BigVGAN must be frozen"
            assert not any(p.requires_grad for p in muq.model.parameters()), "MuQ must be frozen"
            assert not muq.model.training and not bigvgan.training, "MuQ/BigVGAN must be in eval mode"
        stages["load_models"] = {"status": "ok"}

        with stage("load_data"):
            crops = load_mel_crops(REPO_ROOT / cfg["manifests"][args.n], h, cfg["num_mel_hops"])
            ids = [c.track_id for c in crops]
        stages["load_data"] = {"status": "ok", "crops": ids}

        with stage("muq_forward"):
            feats = torch.cat([muq.encode(c.waveform.to(device)).clone() for c in crops])
            finite("MuQ last_hidden_state", feats)
            assert feats.device.type == device.type, feats.device
        stages["muq_forward"] = {"status": "ok", "shape": list(feats.shape), "device": str(feats.device)}

        targets = torch.cat([c.target_mel for c in crops]).to(device)
        t_muq, t_mel = feats.shape[1], targets.shape[-1]
        with stage("time_alignment"):
            muq_centers = muq_token_centers(muq.model.model, t_muq)
            mel_centers = bigvgan_mel_frame_centers(h, t_mel)
            positions = source_positions(mel_centers, muq_centers).to(device)
            assert positions.shape == (t_mel,) and float(positions.min()) >= 0 and float(positions.max()) <= t_muq - 1
            assert muq_centers[1] - muq_centers[0] == 480 * 2, "MuQ token spacing changed"
        stages["time_alignment"] = {
            "status": "ok", "mode": cfg.get("temporal_alignment"),
            "muq_token_center_first_spacing": [float(muq_centers[0]), float(muq_centers[1] - muq_centers[0])],
            "mel_frame_center_first_spacing": [float(mel_centers[0]), float(mel_centers[1] - mel_centers[0])],
            "positions_first_last": [float(positions[0]), float(positions[-1])],
        }
        if cfg.get("temporal_alignment") != "frame_centers":
            stages["time_alignment"]["note"] = "config uses length_resize; corrected positions computed but unused"
            positions = None
        del muq

        model = MuQToMelTranslator(feats.shape[-1], cfg["hidden_channels"], h.num_mels).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["adam_betas"]))
        losses = []
        with stage("translator_forward_backward_optimizer_step"):
            for _ in range(args.steps):
                model.train()
                pred = model(feats, t_mel, positions)
                assert pred.shape == targets.shape, (pred.shape, targets.shape)
                loss = (pred - targets).abs().mean()
                finite("loss", loss)
                opt.zero_grad()
                loss.backward()
                grads = [p.grad for p in model.parameters()]
                assert all(g is not None for g in grads), "missing gradient"
                finite("gradients", torch.cat([g.flatten() for g in grads]))
                before = next(model.parameters()).detach().clone()
                opt.step()
                assert not torch.equal(before, next(model.parameters()).detach()), "optimizer step changed nothing"
                losses.append(loss.item())
            synchronize(device)
        stages["translator_forward_backward_optimizer_step"] = {"status": "ok", "losses": losses}

        mel_floor = bigvgan_log_mel_floor()
        model.eval()
        with torch.inference_mode(), stage("raw_predicted_mel"):
            raw = model(feats, t_mel, positions)
            finite("raw predicted mel", raw)
        stages["raw_predicted_mel"] = {"status": "ok", "shape": list(raw.shape), "mel_l1": mel_l1(raw, targets),
                                       "fraction_below_floor": float((raw < mel_floor).float().mean())}

        with torch.inference_mode(), stage("decode_mel_floor_clamp"):
            clamped, n_clamped = clamp_to_mel_floor(raw, mel_floor)
            assert float(clamped.min()) >= mel_floor and torch.equal(clamped[raw >= mel_floor], raw[raw >= mel_floor])
        stages["decode_mel_floor_clamp"] = {"status": "ok", "floor": mel_floor, "bins_clamped": n_clamped,
                                            "enabled_in_config": bool(cfg.get("decode_clamp_to_mel_floor"))}

        with torch.inference_mode(), stage("bigvgan_inference"):
            oracle = bigvgan(targets)
            audio = bigvgan(clamped if cfg.get("decode_clamp_to_mel_floor") else raw)
            finite("BigVGAN output", audio)
            assert audio.shape == oracle.shape == (len(crops), 1, t_mel * h.hop_size), audio.shape
        stages["bigvgan_inference"] = {"status": "ok", "shape": list(audio.shape),
                                       "peak": float(audio.abs().max()), "oracle_peak": float(oracle.abs().max())}

        ckpt = out / "smoke_checkpoint.pt"
        with stage("checkpoint_save_load"):
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                        "device_metadata": info, "seeding": seeding}, ckpt)
            state = load_portable(ckpt)
            assert all(v.device.type == "cpu" for v in state["model_state_dict"].values())
            on_cpu = MuQToMelTranslator(feats.shape[-1], cfg["hidden_channels"], h.num_mels)
            on_cpu.load_state_dict(state["model_state_dict"])
            on_dev = MuQToMelTranslator(feats.shape[-1], cfg["hidden_channels"], h.num_mels).to(device)
            on_dev.load_state_dict(state["model_state_dict"])
            opt2 = torch.optim.Adam(on_dev.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["adam_betas"]))
            opt2.load_state_dict(state["optimizer_state_dict"])
            with torch.inference_mode():
                same_dev = (on_dev.eval()(feats, t_mel, positions) - raw).abs().max().item()
                cpu_pos = positions.cpu() if positions is not None else None
                cross = (on_cpu.eval()(feats.cpu(), t_mel, cpu_pos) - raw.cpu()).abs().max().item()
            assert same_dev == 0.0, f"reloaded model differs on the same device by {same_dev}"
            assert cross < 1e-3, f"CPU vs {device} prediction differs by {cross}"
        stages["checkpoint_save_load"] = {"status": "ok", "path": str(ckpt.relative_to(REPO_ROOT)),
                                          "max_abs_diff_same_device": same_dev,
                                          "max_abs_diff_cpu_vs_device": cross}

        with torch.inference_mode(), stage("shuffled_across_control"):
            pairing = shuffled_across_pairing(ids)
            idx = torch.tensor([ids.index(pairing[i]) for i in ids], device=device)
            shuf = model(feats[idx], t_mel, positions)
            shuf_audio = bigvgan(clamp_to_mel_floor(shuf, mel_floor)[0] if cfg.get("decode_clamp_to_mel_floor") else shuf)
            finite("shuffled-across audio", shuf_audio)
        stages["shuffled_across_control"] = {"status": "ok", "pairing": pairing,
                                             "mel_l1_shuffled": mel_l1(shuf, targets), "mel_l1_correct": mel_l1(raw, targets)}
        report["passed"] = True
    except Exception as exc:  # noqa: BLE001 -- any stage failure is recorded in the report, then exit 1
        failed = next((k for k, v in stages.items() if v.get("status") == "running"), "setup")
        stages.setdefault(failed, {})["status"] = "failed"
        stages[failed]["error"] = f"{type(exc).__name__}: {exc}"
        stages[failed]["notes"] = list(getattr(exc, "__notes__", []))
        report["passed"] = False
        print(f"FAILED at stage {failed!r} on {device}: {type(exc).__name__}: {exc}")
        for note in getattr(exc, "__notes__", []):
            print(f"  {note}")
    report["seconds"] = time.perf_counter() - t_start
    report["device_memory"] = memory_stats(device)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nbackend smoke test on {device} ({info['device_name']}): {'PASSED' if report['passed'] else 'FAILED'} "
          f"in {report['seconds']:.1f}s -> {(out / 'report.json').relative_to(REPO_ROOT)}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
