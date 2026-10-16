"""TASK-15: small-N memorization test of

    audio -> frozen MuQ last_hidden_state -> MuQToMelTranslator -> BigVGAN-v2 log-mel
          -> frozen BigVGAN-v2 -> audio

Mechanical viability only: N=2 (and N=8 if N=2 passes) fixed crops, training and evaluation on
the same crops. Not a generalization experiment and not evidence of MuQ invertibility.

    uv run python scripts/train_task15_bigvgan_memorization.py --n 2
    uv run python scripts/train_task15_bigvgan_memorization.py --n 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml

import wandb
from muq_inversion.bigvgan_common import (
    BIGVGAN_DIR,
    audio_metrics,
    load_bigvgan,
    load_mel_crops,
    mel_l1,
    shuffled_across_pairing,
)
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.logging_utils import capture_environment_manifest, select_device
from muq_inversion.models.losses import MultiResolutionSTFTLoss
from muq_inversion.models.muq_mel_translator import MuQToMelTranslator
from muq_inversion.wandb_logging import init_run, log_artifact_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/task15_bigvgan_memorization.yaml"
STFT = ((512, 1024, 2048), (50, 120, 240), (240, 600, 1200))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_atomic(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def plot_mels(path: Path, title: str, target: torch.Tensor, pred: torch.Tensor, shuffled: torch.Tensor) -> None:
    panels = [
        ("target (official mel)", target, {"vmin": -11.5, "vmax": 2.5, "cmap": "magma"}),
        ("predicted", pred, {"vmin": -11.5, "vmax": 2.5, "cmap": "magma"}),
        ("predicted - target", pred - target, {"vmin": -2, "vmax": 2, "cmap": "RdBu_r"}),
        ("shuffled-across predicted", shuffled, {"vmin": -11.5, "vmax": 2.5, "cmap": "magma"}),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.2), sharey=True)
    for ax, (name, mel, kw) in zip(axes, panels, strict=True):
        im = ax.imshow(mel.squeeze().detach().cpu().numpy(), origin="lower", aspect="auto", **kw)
        ax.set_title(name)
        ax.set_xlabel("mel frame (10.7 ms)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("mel band")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", choices=["2", "8"], required=True, help="number of fixed crops (TASK-15 caps at 8)")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    n = args.n

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = select_device(cfg["device_request"], cfg["allow_cpu_fallback"])
    environment = capture_environment_manifest(REPO_ROOT)
    run_root = REPO_ROOT / "runs" / cfg["run_dir_name"] / f"n{n}"
    out = REPO_ROOT / "artifacts" / cfg["artifact_dir_name"] / f"n{n}"
    out.mkdir(parents=True, exist_ok=True)

    bigvgan = load_bigvgan(device)
    h = bigvgan.h
    manifest = REPO_ROOT / cfg["manifests"][n]
    crops = load_mel_crops(manifest, h, cfg["num_mel_hops"])
    assert len(crops) == int(n), f"expected {n} crops, got {len(crops)}"
    ids = [c.track_id for c in crops]

    muq = FrozenMuQEncoder(cfg["muq_model_id"], device)
    muq_feats = torch.cat([muq.encode(c.waveform.to(device)).clone() for c in crops])  # (N, T_muq, 1024)
    del muq
    targets = torch.cat([c.target_mel for c in crops]).to(device)  # (N, 100, T_mel)
    if not (torch.isfinite(muq_feats).all() and torch.isfinite(targets).all()):
        raise SystemExit("STOP CONDITION: non-finite MuQ features or target mel")
    t_muq, t_mel = muq_feats.shape[1], targets.shape[-1]

    model = MuQToMelTranslator(muq_feats.shape[-1], cfg["hidden_channels"], h.num_mels).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["adam_betas"]))
    pairing = shuffled_across_pairing(ids)
    shuffled_idx = torch.tensor([ids.index(pairing[i]) for i in ids], device=device)
    geometry = {
        "crop_samples": cfg["num_mel_hops"] * h.hop_size,
        "crop_seconds": cfg["num_mel_hops"] * h.hop_size / h.sampling_rate,
        "muq_frames": t_muq,
        "muq_frame_rate_hz": t_muq / (cfg["num_mel_hops"] * h.hop_size / h.sampling_rate),
        "mel_frames": t_mel,
        "mel_frame_rate_hz": h.sampling_rate / h.hop_size,
        "mel_bands": h.num_mels,
        "temporal_conversion": f"linear interpolation {t_muq} -> {t_mel} frames after Linear(1024->{cfg['hidden_channels']})",
        "translator_parameters": params,
        "receptive_field_mel_frames": model.receptive_field_mel_frames,
        "bigvgan_parameters": sum(p.numel() for p in bigvgan.parameters()),
    }
    print(json.dumps(geometry, indent=2))
    print(f"shuffled-across pairing: {pairing}")

    resolved = {
        **cfg, "n": int(n), "manifest_hash": file_hash(manifest),
        "bigvgan_config_hash": file_hash(BIGVGAN_DIR / "config.json"),
        "bigvgan_generator_hash": file_hash(BIGVGAN_DIR / "bigvgan_generator.pt"),
        "uv_lock_hash": file_hash(REPO_ROOT / "uv.lock"), "device": str(device),
        "geometry": geometry, "crops": ids, "shuffled_across_pairing": pairing,
        "bigvgan_mel_hparams": {k: h[k] for k in ("sampling_rate", "num_mels", "n_fft", "hop_size", "win_size", "fmin", "fmax")},
    }
    run = init_run(
        config=resolved, environment=environment, run_name=f"task15_bigvgan_n{n}", mode=args.wandb_mode,
        group=cfg["wandb_group"], job_type="memorization", tags=[*cfg["wandb_tags"], f"n{n}"],
    )

    # Oracle reconstructions: fixed for the whole run.
    with torch.inference_mode():
        oracle_audio = [bigvgan(c.target_mel.to(device)).squeeze().cpu() for c in crops]

    t0 = time.perf_counter()
    train_loss_history: list[float] = []
    best = {"mel_l1": float("inf"), "step": 0, "state": None}
    for step in range(1, cfg["steps"] + 1):
        model.train()
        pred = model(muq_feats, t_mel)
        loss = (pred - targets).abs().mean()
        if not torch.isfinite(loss):
            raise SystemExit(f"STOP CONDITION: non-finite loss at step {step}")
        opt.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))  # measure only
        opt.step()
        train_loss_history.append(loss.item())
        log = {"train/l1_log_mel": loss.item(), "train/grad_norm": grad_norm}
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            model.eval()
            with torch.inference_mode():
                pred = model(muq_feats, t_mel)
                pred_shuf = model(muq_feats[shuffled_idx], t_mel)
                corr = [
                    audio_metrics(oracle_audio[i], bigvgan(pred[i : i + 1]).squeeze().cpu(), h,
                                  lambda a, b: torch.zeros(1))["waveform_correlation"]
                    for i in range(len(crops))
                ]
            eval_mel_l1 = mel_l1(pred, targets)
            log |= {
                "eval/mel_l1_correct": eval_mel_l1,
                "eval/mel_l1_shuffled_across": mel_l1(pred_shuf, targets),
                "eval/pred_vs_oracle_waveform_corr_mean": float(np.mean(corr)),
                "eval/pred_vs_oracle_waveform_corr_min": float(np.min(corr)),
            }
            # Memorization regime: the evaluation crops ARE the training crops, so selecting the
            # best-memorized state by this metric involves no held-out data.
            if eval_mel_l1 < best["mel_l1"]:
                best = {"mel_l1": eval_mel_l1, "step": step,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            log["eval/best_mel_l1_so_far"] = best["mel_l1"]
            print(f"step {step}: loss {loss.item():.4f}  mel L1 correct {eval_mel_l1:.4f}  "
                  f"shuffled {log['eval/mel_l1_shuffled_across']:.4f}  pred-vs-oracle corr mean {np.mean(corr):.3f} min {np.min(corr):.3f}")
        run.log(log, step=step)
    train_seconds = time.perf_counter() - t0

    def save_checkpoint(name: str, state: dict, step: int) -> Path:
        path = run_root / "checkpoints" / f"{name}.pt"
        save_atomic(path, {"step": step, "model_state_dict": state, "optimizer_state_dict": opt.state_dict() if name == "final" else None,
                           "resolved_config": resolved, "git_commit": environment.get("git_commit"),
                           "git_dirty": environment.get("git_dirty"), "wandb_run_id": run.id})
        check = MuQToMelTranslator(muq_feats.shape[-1], cfg["hidden_channels"], h.num_mels).to(device)
        check.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model_state_dict"])
        return path

    save_checkpoint("final", model.state_dict(), cfg["steps"])
    save_checkpoint("best", best["state"], best["step"])

    stft = MultiResolutionSTFTLoss(*STFT)
    crit = cfg["criteria"]

    def evaluate(state: dict, label: str, write_outputs: bool) -> tuple[dict, dict]:
        m = MuQToMelTranslator(muq_feats.shape[-1], cfg["hidden_channels"], h.num_mels).to(device)
        m.load_state_dict(state)
        m.eval()
        with torch.inference_mode():
            pred = m(muq_feats, t_mel).cpu()
            pred_shuf = m(muq_feats[shuffled_idx], t_mel).cpu()
        per_crop = {}
        for i, c in enumerate(crops):
            with torch.inference_mode():
                pred_audio = bigvgan(pred[i : i + 1].to(device)).squeeze().cpu()
                shuf_audio = bigvgan(pred_shuf[i : i + 1].to(device)).squeeze().cpu()
            original = c.waveform.squeeze()
            oracle_m = audio_metrics(original, oracle_audio[i], h, stft)
            pred_m = audio_metrics(original, pred_audio, h, stft)
            shuf_m = audio_metrics(original, shuf_audio, h, stft)
            pred_vs_oracle = audio_metrics(oracle_audio[i], pred_audio, h, stft)
            r = {
                "mel_l1_pred_vs_target": mel_l1(pred[i], c.target_mel),
                "mel_l1_shuffled_vs_target": mel_l1(pred_shuf[i], c.target_mel),
                "shuffled_source": pairing[c.track_id],
                "oracle_vs_original": oracle_m, "pred_vs_original": pred_m, "shuffled_vs_original": shuf_m,
                "pred_vs_oracle": pred_vs_oracle,
            }
            r["criteria"] = {
                "1_mel_l1_within_oracle_roundtrip": r["mel_l1_pred_vs_target"] <= oracle_m["resynthesis_mel_l1"],
                "2_pred_vs_oracle_corr": pred_vs_oracle["waveform_correlation"] >= crit["pred_vs_oracle_waveform_corr_min"],
                "3_resynthesis_mel_l1_ratio": pred_m["resynthesis_mel_l1"] <= crit["resynthesis_mel_l1_ratio_max"] * oracle_m["resynthesis_mel_l1"],
                "4_shuffled_control_separates": r["mel_l1_shuffled_vs_target"] >= crit["shuffled_mel_l1_ratio_min"] * r["mel_l1_pred_vs_target"],
            }
            per_crop[c.track_id] = r
            if write_outputs:
                sr = h.sampling_rate
                sf.write(out / f"{c.track_id}_original.wav", original.numpy(), sr)
                sf.write(out / f"{c.track_id}_bigvgan_oracle.wav", oracle_audio[i].numpy(), sr)
                sf.write(out / f"{c.track_id}_predicted.wav", pred_audio.numpy(), sr)
                sf.write(out / f"{c.track_id}_shuffled_across.wav", shuf_audio.numpy(), sr)
                png = out / f"{c.track_id}_mel_comparison.png"
                plot_mels(png, f"{c.track_id} (N={n}, {label} checkpoint) -- mel L1 pred {r['mel_l1_pred_vs_target']:.3f}, "
                          f"shuffled {r['mel_l1_shuffled_vs_target']:.3f}", c.target_mel, pred[i], pred_shuf[i])
                run.log({f"mel/{c.track_id}": wandb.Image(str(png)),
                         **{f"audio/{c.track_id}_{k}": wandb.Audio(v.numpy(), sample_rate=sr)
                            for k, v in (("original", original), ("oracle", oracle_audio[i]),
                                         ("predicted", pred_audio), ("shuffled_across", shuf_audio))}})
            print(f"[{label}] {c.track_id:>18}: mel L1 pred {r['mel_l1_pred_vs_target']:.3f} (oracle round-trip "
                  f"{oracle_m['resynthesis_mel_l1']:.3f}) shuffled {r['mel_l1_shuffled_vs_target']:.3f} | pred-vs-oracle corr "
                  f"{pred_vs_oracle['waveform_correlation']:.3f} | resynth mel L1 pred {pred_m['resynthesis_mel_l1']:.3f} | "
                  f"STFT pred {pred_m['stft_distance']:.3f} oracle {oracle_m['stft_distance']:.3f} shuffled "
                  f"{shuf_m['stft_distance']:.3f} | {'PASS' if all(r['criteria'].values()) else 'FAIL'}")

        def mean(f):
            return float(np.mean([f(r) for r in per_crop.values()]))

        summary = {
            "mel_l1_pred_vs_target": mean(lambda r: r["mel_l1_pred_vs_target"]),
            "mel_l1_shuffled_vs_target": mean(lambda r: r["mel_l1_shuffled_vs_target"]),
            "oracle_resynthesis_mel_l1": mean(lambda r: r["oracle_vs_original"]["resynthesis_mel_l1"]),
            "pred_resynthesis_mel_l1": mean(lambda r: r["pred_vs_original"]["resynthesis_mel_l1"]),
            "shuffled_resynthesis_mel_l1": mean(lambda r: r["shuffled_vs_original"]["resynthesis_mel_l1"]),
            "oracle_stft": mean(lambda r: r["oracle_vs_original"]["stft_distance"]),
            "pred_stft": mean(lambda r: r["pred_vs_original"]["stft_distance"]),
            "shuffled_stft": mean(lambda r: r["shuffled_vs_original"]["stft_distance"]),
            "pred_vs_oracle_waveform_corr": mean(lambda r: r["pred_vs_oracle"]["waveform_correlation"]),
            "memorization_criteria_passed": all(all(r["criteria"].values()) for r in per_crop.values()),
        }
        run.log({f"{label}/{k}": float(v) for k, v in summary.items()})
        return summary, per_crop

    best_summary, best_per_crop = evaluate(best["state"], "best", write_outputs=True)
    final_summary, final_per_crop = evaluate(model.state_dict(), "final", write_outputs=False)
    losses = np.array(train_loss_history)
    report = {
        "task": "TASK-15", "n": int(n), "geometry": geometry, "train_seconds": train_seconds,
        "evaluation_protocol": (
            "Primary: best checkpoint by eval mel L1 on the (training = evaluation) crops, chosen among evals every "
            f"{cfg['eval_every']} steps. Added after the first N=2 run, whose final step (5000) landed on a transient "
            "Adam/L1 loss spike (0.025 -> 0.41) after meeting every criterion at steps 3250-4750. Final-step results "
            "are reported alongside."
        ),
        "best_checkpoint_step": best["step"],
        "best": {"summary": best_summary, "per_crop": best_per_crop},
        "final": {"summary": final_summary, "per_crop": final_per_crop},
        "train_loss": {"min": float(losses.min()), "last": float(losses[-1]),
                       "steps_above_0p1_after_step_1000": int((losses[1000:] > 0.1).sum()),
                       "history": losses.round(5).tolist()},
        "resolved_config": resolved, "wandb_run_id": run.id,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    log_artifact_file(run, out / "report.json", f"task15-report-n{n}", "report")
    run.finish()
    print("best (step", best["step"], ")", json.dumps(best_summary, indent=2))
    print("final", json.dumps(final_summary, indent=2))
    print(f"\nN={n}: memorization criteria at best checkpoint (step {best['step']}): "
          f"{'PASSED' if best_summary['memorization_criteria_passed'] else 'NOT PASSED'}; at final step: "
          f"{'PASSED' if final_summary['memorization_criteria_passed'] else 'NOT PASSED'}. Listening still required.")


if __name__ == "__main__":
    main()
