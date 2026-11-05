"""TASK-23: one-crop decoder-overfit diagnostic -- single entry point (tasks/TASK-23.md).

    uv run python scripts/task23_overfit.py --config configs/task23.yaml --device xpu --run-name task23_v1
    uv run python scripts/task23_overfit.py --config configs/task23_smoke.yaml --device mps --run-name smoke_x

Stages (resumable and idempotent; a finished run is never overwritten):
  1. pair: one fixed TRAINING crop -> frozen MuQ first-block features (hook-checked, the exact
     TASK-22 cached tensor when `feature_source: task22_cache`), the TASK-22 target mel (official
     BigVGAN mel of the peak-normalized track's crop), the valid mask, the waveform, and the mel-frame
     source positions; verification of identity, alignment, normalization, and BigVGAN compatibility;
     cached with tensor hashes (runs/task23/<run>/pair.pt; metadata in artifacts/.../pair/).
  2. arms A (free tensor), B (TASK-22 recipe), C (B's init, regularizers disabled), and C_seed1 only
     if C misses the criterion. B/C: 20,000 updates, extended to 50,000 at LR/10 if the
     evaluation-mode L1 is still above 1% of its initial value. Train- and eval-mode metrics every 100.
  3. plots, BigVGAN audio, report.json and TASK-23_RESULTS.md.
SIGTERM/SIGINT: latest.pt is saved at the end of the current update and the script exits 3;
rerunning with the same --run-name resumes.
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
import soundfile as sf
import torch

from muq_inversion.audio import load_resample_validate
from muq_inversion.bigvgan_common import (
    BIGVGAN_DIR,
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
    ArmSpec,
    FreeMel,
    atomic_write_text,
    boundary_frames,
    disable_regularizers,
    load_pair,
    masked_l1,
    regularizer_audit,
    save_pair,
    state_dict_sha256,
    train_arm,
)
from muq_inversion.time_alignment import (
    bigvgan_mel_frame_centers,
    muq_token_centers,
    source_positions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_task18_heldout import load_config

DB_PER_NAT = 20 / np.log(10)  # log-magnitude (natural log) -> dB
# Training-loop features of TASK-22 (scripts/train_task18_heldout.py), verified by reading the code.
RECIPE_LOOP_FACTS = {
    "gradient_clipping": "none (clip_grad_norm_(..., inf) only measures the norm)",
    "input_noise": "none", "augmentation": "none (cached crops, no random crop)",
    "lr_schedule": "constant", "mixed_precision": "none (FP32, no autocast, no grad scaler)",
    "feature_normalization": "none (raw MuQ features into the input Linear)",
    "target_normalization": "none (raw BigVGAN log-mel); decode-time clamp to the log floor only",
    "loss": "(pred - mel).abs().mean() over all bins and frames (no padding, so no mask)",
}


def resolve_recipe(cfg: dict) -> dict:
    """TASK-22's resolved arm config, checked against what this diagnostic assumes."""
    base = load_config(REPO_ROOT / cfg["recipe"]["config"])
    r = {**base, **base["arms"][cfg["recipe"]["arm"]]}
    expect = {"muq_layer": {"block": 1}, "temporal_alignment": "frame_centers", "decode_clamp_to_mel_floor": True,
              "lr_schedule": {"type": "constant"}}
    bad = {k: r.get(k) for k, v in expect.items() if r.get(k) != v}
    if bad:
        raise SystemExit(f"STOP: TASK-22 recipe differs from what TASK-23 assumes: {bad}")
    return r


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --- stage 1: the fixed pair -------------------------------------------------------------

def selection_scores(rows: list[dict], sr: int) -> list[dict]:
    """Harmonic energy fraction (HPSS) and onset-strength peak/median of each candidate crop."""
    out, tracks = [], {}
    for r in rows:
        if r["path"] not in tracks:
            t, _ = load_resample_validate(REPO_ROOT / r["path"], sr)
            tracks[r["path"]] = librosa.util.normalize(t) * PEAK_NORMALIZE_TARGET
        s = int(r["crop_start_sample"])
        y = tracks[r["path"]][s:s + int(r["crop_num_samples"])]
        harm, perc = librosa.effects.hpss(y)
        on = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
        out.append({"crop_id": r["crop_id"], "start": s, "harmonic_fraction": float((harm ** 2).sum() / ((harm ** 2).sum() + (perc ** 2).sum())),
                    "onset_peak_over_median": float(on.max() / (np.median(on) + 1e-9)), "rms": float(np.sqrt((y ** 2).mean()))})
    from scipy.stats import rankdata
    score = rankdata([o["harmonic_fraction"] for o in out]) + rankdata([o["onset_peak_over_median"] for o in out])
    for o, sc in zip(out, score, strict=True):
        o["rank_sum"] = float(sc)
    return sorted(out, key=lambda o: -o["rank_sum"])


def find_task22_cache(cache_dir: Path, key: tuple[str, int]) -> tuple[Path, dict]:
    hits = []
    for p in sorted(cache_dir.glob("task18_v1_L1_*.pt")):
        d = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        if d.get("key_parts", {}).get("muq_layer") == "conformer_block_1_output" and key in d["keys"]:
            hits.append((p, d))
    if not hits:
        raise SystemExit(f"STOP: no TASK-22 first-block cache in {cache_dir} contains {key}")
    return hits[0]


def regions(target: torch.Tensor, wave: np.ndarray, sr: int, hop: int, width: int, max_bin: int) -> dict:
    """Transient = strongest onset frame; harmonic region = the `width`-frame window with the
    highest mean harmonic fraction of the (linear) mel in bins < max_bin (HPSS on exp(log-mel))."""
    on = librosa.onset.onset_strength(y=wave, sr=sr, hop_length=hop, center=False)
    t_frames = target.shape[-1]
    tf = int(np.argmax(on))
    mag = np.exp(target[0].numpy())[:max_bin]
    harm, perc = librosa.decompose.hpss(mag)
    frac = harm.sum(0) / (harm.sum(0) + perc.sum(0) + 1e-12)
    w = min(width, t_frames)
    smooth = np.convolve(frac, np.ones(w) / w, mode="valid")
    h0 = int(np.argmax(smooth))
    t0 = int(np.clip(tf - w // 2, 0, t_frames - w))
    return {"transient_frame": tf, "transient_window": [t0, t0 + w], "harmonic_window": [h0, h0 + w],
            "harmonic_bins": [0, max_bin], "harmonic_fraction_window_mean": float(smooth[h0]),
            "onset_note": "librosa onset_strength(center=False, hop 256): frame j covers samples from 256*j, "
                          "about half a hop from mel frame j's center; used only to place the zoom window"}


def prepare_pair(cfg: dict, recipe: dict, device: torch.device, pair_path: Path, pair_art: Path) -> dict:
    if pair_path.exists():
        d = load_pair(pair_path)
        print(f"pair cache hit (hashes verified): {pair_path.relative_to(REPO_ROOT)}")
        return d
    pc = cfg["pair"]
    rows = read_csv(REPO_ROOT / pc["crop_manifest"])
    row = next((r for r in rows if r["crop_id"] == pc["crop_id"]), None)
    if row is None or row["role"] != "train_pool" or row["split"] != "train":
        raise SystemExit(f"STOP: {pc['crop_id']} is not a training-pool crop of a training track")
    track_row = next(t for t in read_csv(REPO_ROOT / pc["track_manifest"]) if t["track_id"] == row["track_id"])
    if file_sha256(REPO_ROOT / row["path"]) != track_row["checksum_sha256"] or track_row["split"] != "train":
        raise SystemExit(f"STOP: {row['path']} checksum or split mismatch")
    bigvgan = load_bigvgan(device)
    h = bigvgan.h
    n_samples, t_mel = recipe["num_mel_hops"] * recipe["hop_size"], recipe["num_mel_hops"]
    if int(row["crop_num_samples"]) != n_samples or h.hop_size != recipe["hop_size"]:
        raise SystemExit("STOP: crop length / hop differs from TASK-22")

    # Selection evidence: the same scoring over the memorized training crops (what the config cites).
    cands = [r for r in rows if r["eval_set"] == "memorized"]
    scores = selection_scores(cands, h.sampling_rate)
    chosen_memorized = f"{row['track_id']}:eval:memorized:{row['crop_id'].rsplit(':', 1)[1]}"
    top = scores[0]["crop_id"]

    # Waveform and target exactly as build_features (TASK-22): resample, peak-normalize the whole
    # track to 0.95, crop; official BigVGAN mel on CPU.
    track, orig_sr = load_resample_validate(REPO_ROOT / row["path"], h.sampling_rate)
    track_peak = float(np.abs(track).max())
    track = librosa.util.normalize(track) * PEAK_NORMALIZE_TARGET
    s = int(row["crop_start_sample"])
    if s + n_samples > len(track):
        raise SystemExit("STOP: crop exceeds the track")
    wave = torch.from_numpy(np.ascontiguousarray(track[s:s + n_samples])).float()
    mel = official_mel(wave[None], h)
    if mel.shape != (1, h.num_mels, t_mel):
        raise SystemExit(f"STOP: target mel {tuple(mel.shape)}")

    # Frozen MuQ, first block, hook-checked; determinism over two passes; MuQ unchanged.
    muq = FrozenMuQEncoder(recipe["muq_model_id"], device)
    inner = muq.model.model
    layer = resolve_muq_layer(recipe["muq_layer"], len(inner.conformer.layers))
    frozen = {"eval_mode": not muq.model.training, "all_requires_grad_false": all(not p.requires_grad for p in muq.model.parameters())}
    if not all(frozen.values()):
        raise SystemExit(f"STOP: MuQ not frozen: {frozen}")
    with torch.inference_mode():
        psum0 = torch.stack([torch.stack([p.sum(), p.abs().sum()]) for p in muq.model.parameters()]).cpu()  # FP32: MPS has no float64
    fresh, check = muq.encode_block_output(wave[None].to(device), layer["block"])
    fresh2, _ = muq.encode_block_output(wave[None].to(device), layer["block"])
    fresh, fresh2 = fresh.float().cpu().clone(), fresh2.float().cpu().clone()
    last = muq.encode(wave[None].to(device)).float().cpu()
    with torch.inference_mode():
        psum1 = torch.stack([torch.stack([p.sum(), p.abs().sum()]) for p in muq.model.parameters()]).cpu()  # FP32: MPS has no float64
    frozen["parameters_unchanged"] = bool(torch.equal(psum0, psum1))
    frozen["still_eval_mode"] = not muq.model.training
    t_muq = fresh.shape[1]
    tok = muq_token_centers(inner, t_muq)
    del muq

    feature_info = {"layer": layer, "index_check": check, "fresh_bit_exact_two_passes": bool(torch.equal(fresh, fresh2)),
                    "fresh_vs_last_hidden_state_cosine_mean": float(torch.nn.functional.cosine_similarity(fresh, last, dim=-1).mean()),
                    "source": pc["feature_source"]}
    features = fresh
    if pc["feature_source"] == "task22_cache":
        cache_path, cache = find_task22_cache(REPO_ROOT / pc["task22_cache_dir"], (row["path"], s))
        i = cache["keys"].index((row["path"], s))
        cached = {k: cache["tensors"][k][i].clone() for k in ("muq", "mel", "wave")}
        diff = float((cached["muq"][None] - fresh).abs().max())
        mel_diff = float((cached["mel"][None] - mel).abs().max())
        feature_info |= {"task22_cache": str(cache_path.relative_to(REPO_ROOT)), "task22_cache_index": i,
                         "task22_cache_key_parts": cache["key_parts"], "task22_cache_layer_check": cache.get("layer_check"),
                         "fresh_vs_task22_cache_max_abs_diff": diff,
                         "task22_cache_mel_equal_recomputed": bool(torch.equal(cached["mel"][None], mel)),
                         "task22_cache_mel_vs_recomputed_max_abs_diff": mel_diff,
                         "task22_cache_wave_equal_recomputed": bool(torch.equal(cached["wave"], wave))}
        del cache
        # The waveform must match bit for bit (same crop). The mel is recomputed on CPU in a batch of 1; TASK-22
        # computed it in batches of 64 on another node, so CPU/BLAS rounding may differ: tolerated up to
        # mel_match_atol, and the CACHED mel (the exact TASK-22 training target) is the one used.
        if not feature_info["task22_cache_wave_equal_recomputed"]:
            raise SystemExit(f"STOP: TASK-22 cached waveform differs from the recomputed one: {feature_info}")
        if mel_diff > pc["mel_match_atol"]:
            raise SystemExit(f"STOP: TASK-22 cached target mel differs from the recomputed one by {mel_diff} > {pc['mel_match_atol']}")
        mel = cached["mel"][None].float().contiguous()
        feature_info["target_source"] = "task22_cache (exact TASK-22 training target)"
        if diff > pc["feature_match_atol"]:
            raise SystemExit(f"STOP: fresh first-block features differ from TASK-22's cache by {diff} > {pc['feature_match_atol']}")
        features = cached["muq"][None].float()
    elif pc["feature_source"] != "recompute":
        raise SystemExit(f"unknown feature_source {pc['feature_source']}")
    else:
        feature_info["target_source"] = "recomputed (official BigVGAN mel, CPU)"

    # Alignment (physical centers, samples of the 24 kHz crop) and the valid mask.
    mel_c = bigvgan_mel_frame_centers(h, t_mel)
    positions = source_positions(mel_c, tok)
    mask = torch.ones(t_mel, dtype=torch.bool)  # 187 hops of real audio: no padded frame (TASK-22 has no mask)
    rf = MuQToMelTranslator(features.shape[-1], recipe["hidden_channels"], h.num_mels).receptive_field_mel_frames
    edge = (rf - 1) // 2
    boundary = boundary_frames(t_mel, edge, positions, t_muq)
    clamped = ((positions <= 0) | (positions >= t_muq - 1))
    pad = (h.n_fft - h.hop_size) // 2
    alignment = {
        "crop_samples": [s, s + n_samples], "crop_seconds_in_track": [s / h.sampling_rate, (s + n_samples) / h.sampling_rate],
        "mel": {"frames": t_mel, "hop": h.hop_size, "n_fft": h.n_fft, "win_size": h.win_size, "reflect_pad_samples": pad,
                "center": False, "frame_center_samples_first_last": [float(mel_c[0]), float(mel_c[-1])],
                "first_frame_window_samples": [-pad, -pad + h.n_fft], "last_frame_window_samples": [float(mel_c[-1] - h.n_fft / 2), float(mel_c[-1] + h.n_fft / 2)],
                "note": "frames whose n_fft window extends past the crop see reflect-padded audio of the crop, not the track"},
        "muq": {"tokens": t_muq, "token_center_samples_first_last": [float(tok[0]), float(tok[-1])],
                "token_hop_samples": float(np.diff(tok).mean()), "frames_per_second": t_muq / (n_samples / h.sampling_rate),
                "extraction": "isolated crop (47,872 samples), as TASK-22; tokens near the crop ends see only crop audio"},
        "source_positions_min_max": [float(positions.min()), float(positions.max())],
        "edge_replicated_mel_frames": {"count": int(clamped.sum()), "frames": clamped.nonzero().flatten().tolist()},
        "translator_receptive_field_mel_frames": rf,
        "boundary_definition": (f"mel frames within {edge} of either crop end (half the translator's {rf}-frame receptive "
                                "field: its convolutions zero-pad there) or with an edge-replicated MuQ source position"),
        "boundary_frames": int(boundary.sum()), "interior_frames": int((~boundary).sum()),
        "unresolved": [("MuQ token centers are geometric centers of the conv frontend; after the conformer's global "
                       "attention a token is not localized, so a token's 'time' is a convention, not a measured receptive field"),
                       "isolated-crop extraction: MuQ's edge tokens lack the track context outside the crop (as in TASK-22)",
                       ("matching lengths (50 tokens -> 187 frames) does not by itself prove alignment; the frame-center "
                       "mapping was audited in TASK-15/16 (impulse tests in tests/), not re-measured here")],
    }
    floor = bigvgan_log_mel_floor()
    y = mel[0]
    dyn = {"min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()), "std": float(y.std()),
           "p01_p50_p99": [float(q) for q in torch.quantile(y.flatten(), torch.tensor([0.01, 0.5, 0.99]))],
           "range_nats": float(y.max() - y.min()), "range_db": float((y.max() - y.min()) * DB_PER_NAT),
           "fraction_at_log_floor": float((y <= floor + 1e-6).float().mean()), "log_floor": floor}
    if not dyn["range_nats"] > 1.0:
        raise SystemExit(f"STOP: target dynamic range too small: {dyn}")

    # Normalization: identity both ways (verified in code); BigVGAN compatibility of the target.
    clamped_mel = mel.clamp(min=floor)
    with torch.inference_mode():
        y_gt = bigvgan(clamped_mel.to(device)).float().cpu()[0, 0]
    rt = official_mel(y_gt[None], h)
    compat = {"input_shape": list(mel.shape), "num_mels_matches": h.num_mels == mel.shape[1],
              "output_samples": int(y_gt.numel()), "output_matches_crop_length": int(y_gt.numel()) == n_samples,
              "bins_raised_by_decode_clamp": int((mel < floor).sum()), "output_finite": bool(torch.isfinite(y_gt).all()),
              "gt_mel_round_trip_l1": float((rt - mel).abs().mean()),
              "note": "gt_mel_round_trip_l1 = official mel of BigVGAN(target) vs target: a waveform-to-mel round trip, NOT direct mel error"}
    normalization = {"features": "identity (TASK-22 feeds raw features)", "target": "identity (raw log-mel)",
                     "inverse_before_vocoding": "identity, then clamp(min=log floor) exactly as TASK-22's decode",
                     "inverse_check_max_abs": float((clamped_mel - mel).abs().max())}
    reg = regions(mel, wave.numpy(), h.sampling_rate, h.hop_size, pc["zoom_frames"], pc["harmonic_max_bin"])

    tensors = {"features": features.contiguous(), "target_mel": mel.contiguous(), "mask": mask, "waveform": wave, "positions": positions}
    meta = {"task": "TASK-23", "crop": {k: row[k] for k in ("crop_id", "track_id", "genre", "path", "split", "crop_start_sample", "crop_start_seconds", "crop_num_samples")},
            "memorized_alias": chosen_memorized, "track": {"orig_sr": orig_sr, "resampled_samples": len(track), "peak_before_normalize": track_peak,
                                                            "checksum_sha256": track_row["checksum_sha256"]},
            "extraction": {"muq_model_id": recipe["muq_model_id"], "bigvgan_revision": recipe["bigvgan_revision"],
                           "bigvgan_generator_sha256": file_sha256(BIGVGAN_DIR / "bigvgan_generator.pt"),
                           "sample_rate": h.sampling_rate, "peak_normalize": PEAK_NORMALIZE_TARGET, "inference_mode": True,
                           "stochastic": "none (MuQ eval mode, no dropout active; BigVGAN mel deterministic on CPU)",
                           "device": str(device), "torch": torch.__version__},
            "muq_frozen": frozen, "features": feature_info, "alignment": alignment, "target_dynamic_range": dyn,
            "normalization": normalization, "bigvgan_compatibility": compat, "regions": reg,
            "selection_scores": scores, "selection_top_rank_sum": top, "selection_agrees": top == chosen_memorized,
            "shapes": {k: list(v.shape) for k, v in tensors.items()}}
    hashes = save_pair(pair_path, tensors, meta)
    meta["sha256"] = hashes
    pair_art.mkdir(parents=True, exist_ok=True)
    atomic_write_text(pair_art / "pair_meta.json", json.dumps(meta, indent=2, default=str))
    audio = pair_art.parent / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    sf.write(audio / "original.wav", wave.numpy(), h.sampling_rate, subtype="FLOAT")
    sf.write(audio / "ground_truth_mel_bigvgan.wav", y_gt.numpy(), h.sampling_rate, subtype="FLOAT")
    print(f"pair cached: {pair_path.relative_to(REPO_ROOT)} | features {tuple(features.shape)} ({pc['feature_source']}) | "
          f"target {tuple(mel.shape)} range {dyn['range_db']:.1f} dB | GT round trip {compat['gt_mel_round_trip_l1']:.4f} | "
          f"boundary frames {alignment['boundary_frames']} | selection agrees: {meta['selection_agrees']}")
    return load_pair(pair_path)


# --- stage 3: outputs --------------------------------------------------------------------

def spectrogram_figure(target: np.ndarray, pred: np.ndarray, title: str, path: Path,
                       frames: tuple[int, int] | None = None, bins: tuple[int, int] | None = None) -> None:
    """Target and prediction on identical axes and one shared color scale; |error| on its own scale."""
    f0, f1 = frames or (0, target.shape[1])
    b0, b1 = bins or (0, target.shape[0])
    t, p = target[b0:b1, f0:f1], pred[b0:b1, f0:f1]
    err = np.abs(p - t)
    vmin, vmax = float(min(t.min(), p.min())), float(max(t.max(), p.max()))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), sharex=True, sharey=True)
    ext = (f0 - 0.5, f1 - 0.5, b0 - 0.5, b1 - 0.5)
    for a, img, name in ((ax[0], t, "target log-mel"), (ax[1], p, "prediction log-mel")):
        im = a.imshow(img, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="magma", extent=ext)
        a.set_title(name)
    fig.colorbar(im, ax=ax[:2], label="log-mel (nats), shared scale", fraction=0.025)
    im2 = ax[2].imshow(err, origin="lower", aspect="auto", vmin=0, vmax=max(float(err.max()), 1e-6), cmap="viridis", extent=ext)
    ax[2].set_title(f"|error| (mean {err.mean():.4f}, max {err.max():.3f})")
    fig.colorbar(im2, ax=ax[2], label="|error| (nats), separate scale", fraction=0.05)
    for a in ax:
        a.set_xlabel("mel frame (93.75 /s)")
    ax[0].set_ylabel("mel bin")
    fig.suptitle(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def learning_curves(history: list[dict], title: str, path: Path) -> None:
    st = np.array([e["step"] for e in history], dtype=float)
    x = np.maximum(st, 1)
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    panels = [("direct mel L1", ["eval/l1", "train/l1"]), ("interior vs boundary (eval)", ["eval/l1_interior", "eval/l1_boundary"]),
              ("detail metrics (eval, diagnostic only)", ["eval/time_diff_l1", "eval/freq_diff_l1"]),
              ("max |error| (eval)", ["eval/max_abs_err"]), ("gradient norm", ["grad_norm"]), ("learning rate", ["lr"])]
    for a, (name, keys) in zip(ax.flat, panels, strict=True):
        for k in keys:
            y = np.array([np.nan if e.get(k) is None else e[k] for e in history], dtype=float)
            a.plot(x, y, label=k, lw=1)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_title(name)
        a.set_xlabel("update (log; step 0 drawn at 1)")
        a.legend(fontsize=8)
    if history and history[0].get("eval/l1"):
        ax[0, 0].axhline(0.01 * history[0]["eval/l1"], color="k", ls=":", lw=1, label="1% of initial")
        ax[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_wav(path: Path, y: torch.Tensor, sr: int) -> dict:
    """Float WAV, no normalization, no clipping (amplitude errors stay audible/measurable)."""
    a = y.detach().float().cpu().numpy()
    sf.write(path, a, sr, subtype="FLOAT")
    return {"path": str(path.relative_to(REPO_ROOT)), "peak": float(np.abs(a).max()), "rms": float(np.sqrt((a ** 2).mean())),
            "samples_above_1": int((np.abs(a) > 1).sum())}


def arm_outputs(name: str, summary: dict, run_dir: Path, art: Path, pair: dict, bigvgan, h, meta: dict) -> dict:
    target = pair["tensors"]["target_mel"]
    reg = meta["regions"]
    out_dir = art / name
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    learning_curves(hist, f"TASK-23 {name}: learning curves (direct mel metrics before vocoding)", out_dir / "learning_curves.png")
    saved = sorted((run_dir / "preds").glob("step_*.pt"))
    if saved:  # eval-mode prediction at every save step, one shared color scale
        preds = [torch.load(f, map_location="cpu", weights_only=False) for f in saved]
        imgs = [target[0].numpy()] + [q["eval"][0].numpy() for q in preds]
        vmin, vmax = min(float(i.min()) for i in imgs), max(float(i.max()) for i in imgs)
        fig, ax = plt.subplots(1, len(imgs), figsize=(3.2 * len(imgs), 3.4), sharey=True)
        for a, img, lab in zip(ax, imgs, ["target"] + [f"step {q['step']:,}" for q in preds], strict=True):
            im = a.imshow(img, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="magma")
            a.set_title(lab)
            a.set_xlabel("mel frame")
        ax[0].set_ylabel("mel bin")
        fig.colorbar(im, ax=ax, label="log-mel (nats), shared scale", fraction=0.02)
        fig.suptitle(f"TASK-23 {name}: evaluation-mode prediction at each saved step")
        fig.savefig(out_dir / "progression.png", dpi=90, bbox_inches="tight")
        plt.close(fig)
    best = torch.load(run_dir / "preds" / "best.pt", map_location="cpu", weights_only=False)
    final = torch.load(run_dir / "preds" / "final.pt", map_location="cpu", weights_only=False)
    floor = bigvgan_log_mel_floor()
    res = {"audio": {}, "plots": ["learning_curves.png"] + (["progression.png"] if saved else [])}
    for label, p in (("best", best), ("final", final)):
        pred = p["eval"]
        if label == "final" and torch.equal(pred, best["eval"]):
            res["final_equals_best"] = True
            continue
        t_np, p_np = target[0].numpy(), pred[0].numpy()
        tag = f"{label} (step {p['step']})"
        spectrogram_figure(t_np, p_np, f"TASK-23 {name}, {tag}", out_dir / f"{label}_spectrogram.png")
        spectrogram_figure(t_np, p_np, f"{name}, {tag}: transient (onset frame {reg['transient_frame']})",
                           out_dir / f"{label}_zoom_transient.png", frames=tuple(reg["transient_window"]))
        spectrogram_figure(t_np, p_np, f"{name}, {tag}: harmonic region, bins {reg['harmonic_bins']}",
                           out_dir / f"{label}_zoom_harmonic.png", frames=tuple(reg["harmonic_window"]), bins=tuple(reg["harmonic_bins"]))
        res["plots"] += [f"{label}_spectrogram.png", f"{label}_zoom_transient.png", f"{label}_zoom_harmonic.png"]
        mel_in = pred.clamp(min=floor)
        with torch.inference_mode():
            y = bigvgan(mel_in.to(next(bigvgan.parameters()).device)).float().cpu()[0, 0]
        info = write_wav(out_dir / f"{label}_pred_mel_bigvgan.wav", y, h.sampling_rate)
        rt = official_mel(y[None], h)
        info |= {"step": p["step"], "bins_raised_by_decode_clamp": int((pred < floor).sum()),
                 "vocoded_round_trip_l1_vs_target": float((rt - target).abs().mean()),
                 "direct_mel_l1": float(masked_l1(pred, target, pair["tensors"]["mask"]))}
        res["audio"][label] = info
    res.setdefault("final_equals_best", False)
    return res


def interpret(summaries: dict, nondeterminism: dict | None = None) -> dict:
    """The task's interpretation rules, applied mechanically to the provisional criterion."""
    met = {k: v["criterion_met"] for k, v in summaries.items()}
    a, b = met.get("A"), met.get("B")
    c = met.get("C") or met.get("C_seed1", False)
    if not a:
        rule = "A fails: loss/optimization plumbing remains unresolved; B/C are not interpreted."
    elif b and c:
        rule = "B/C succeed: the decoder can memorize this pair. This does not establish transferable reconstruction or invertibility."
    elif not b and c:
        rule = "B fails but C succeeds: regularization or training-mode behavior limits fitting."
    elif b and not c:
        rule = "B succeeds but C fails: unexpected (C only removes regularizers); check determinism before interpreting."
    else:
        rule = ("A succeeds but B/C plateau: decoder parameterization, normalization, upsampling, or optimization needs "
                "investigation; this does not show information loss.")
    if b is not None and met.get("C") is not None and b != met["C"]:
        rule += (" Caution: B and C differ only by run-to-run nondeterminism (identical init and configuration; nothing "
                 "was disabled), so this split is not evidence about regularization.")
    return {"criterion_met": met, "rule": rule}


def write_report(art: Path, report: dict) -> None:
    m, s = report["pair_meta"], report["summaries"]
    lines = [f"# TASK-23 results: one-crop decoder overfit (`{report['run_name']}`)", "",
             (f"Executed on {report['device']['device_name']} ({report['device']['resolved_device']}), commit "
             f"{report['environment'].get('git_commit')} (dirty={report['environment'].get('git_dirty')}), {report['finished_utc']}."),
             "", "## Pair", "",
             (f"- Crop `{m['crop']['crop_id']}` (training track {m['crop']['track_id']}, {float(m['crop']['crop_start_seconds']):.3f} s, "
             f"{m['crop']['crop_num_samples']} samples; = `{m['memorized_alias']}`)."),
             f"- Features {m['shapes']['features']} from `{m['features']['source']}`"
             + (f" (`{m['features'].get('task22_cache')}`; fresh vs cache max |Δ| {m['features'].get('fresh_vs_task22_cache_max_abs_diff', float('nan')):.2e})"
                if m['features']['source'] == 'task22_cache' else "") + f"; {m['features']['layer']['name']} = hidden_states[{m['features']['layer']['hidden_state_index']}], hook-checked.",
             (f"- Target {m['shapes']['target_mel']}, range {m['target_dynamic_range']['range_db']:.1f} dB, "
             f"{100 * m['target_dynamic_range']['fraction_at_log_floor']:.2f}% at the log floor; mask all valid ({m['shapes']['mask'][0]} frames)."),
             (f"- Boundary: {m['alignment']['boundary_definition']} -> {m['alignment']['boundary_frames']} boundary / "
             f"{m['alignment']['interior_frames']} interior frames."),
             (f"- BigVGAN on the ground-truth mel: round-trip mel L1 {m['bigvgan_compatibility']['gt_mel_round_trip_l1']:.4f} "
             "(waveform-to-mel round trip; not comparable to direct mel L1)."),
             "- Unresolved alignment uncertainty: " + "; ".join(m["alignment"]["unresolved"]) + ".", "",
             "## Runs (direct predicted-vs-target mel L1 before vocoding, evaluation mode)", "",
             "| arm | initial | best (step) | final (step) | best / initial | criterion (<=1% of initial) | extension | train-eval max abs diff | interior / boundary at final | time / freq diff at final | runtime |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, v in s.items():
        f = v["final"]
        crit = f"step {v['criterion']['reached_at_step']}" if v["criterion"] else "not met"
        ext = ("-" if not v["extension"] else (f"to {v['spec']['extension_steps']:,}, LR x{v['spec']['extension_lr_factor']:g}"
                                                if v["extension"]["extended"] else "not needed"))
        lines.append(f"| {k} | {v['initial']['eval_l1']:.4f} | {v['best']['eval_l1']:.5f} ({v['best']['step']}) | {f['eval/l1']:.5f} ({f['step']}) | "
                     f"{v['best_relative_to_initial']:.4%} | {crit} | {ext} | {v['max_train_eval_abs_diff']:.1e} | "
                     f"{f['eval/l1_interior']:.5f} / {f['eval/l1_boundary']:.5f} | {f['eval/time_diff_l1']:.5f} / {f['eval/freq_diff_l1']:.5f} | {v['runtime_seconds'] / 60:.1f} min |")
    lines += ["", "Reference: TASK-22's best checkpoint (trained on 512 tracks) had mel L1 0.238 on this crop.", "",
              "## Audio (frozen BigVGAN; float WAV, playback gain 1.0, no per-clip normalization)", ""]
    for k, o in report["outputs"].items():
        for label, a in o["audio"].items():
            lines.append(f"- {k} {label} (step {a['step']}): `{a['path']}` peak {a['peak']:.3f}; vocoded round-trip mel L1 "
                         f"{a['vocoded_round_trip_l1_vs_target']:.4f} (vs ground-truth mel's {m['bigvgan_compatibility']['gt_mel_round_trip_l1']:.4f})")
    lines += ["- `audio/original.wav` and `audio/ground_truth_mel_bigvgan.wav` in this folder.",
              "- Listening: **pending** (no human ratings yet; nothing here is a perceptual claim).", "",
              "## Train/eval mode and regularizers", "",
              f"- Audit (B): {json.dumps(report['regularizers']['B_audit'])}.",
              f"- Disabled in C: {report['regularizers']['C_disabled'] or 'nothing (no dropout, no weight decay, no other explicit regularizer exists in the TASK-22 recipe)'}.",
              f"- Loop facts (TASK-22 code): {'; '.join(f'{k}: {v}' for k, v in RECIPE_LOOP_FACTS.items())}.",
              (f"- B and C final weights identical: {report['b_c_identical']} (same init {report['b_c_same_init']}); "
              f"deterministic algorithms: {report['deterministic_algorithms']}."),
              f"- Nondeterminism (B vs C, eval-mode L1 at shared report points): {json.dumps(report['nondeterminism'])}.", "",
              "## Interpretation", "", f"**Observation (rule applied to the provisional 1% criterion):** {report['interpretation']['rule']}", "",
              "Hypotheses and limits:",
              "- One-pair memorization cannot establish information preservation.",
              "- Success cannot establish correct temporal alignment.",
              "- Failure cannot establish irrecoverable information loss.",
              "- This experiment does not test whether more training data would improve generalization.",
              "- If a predicted mel is nearly exact but its audio is poor, compare with `ground_truth_mel_bigvgan.wav` (vocoder limit).",
              "- The 1% criterion is provisional, not perceptual."]
    atomic_write_text(art / "TASK-23_RESULTS.md", "\n".join(lines) + "\n")


# --- main --------------------------------------------------------------------------------

def arm_spec(name: str, a: dict, cfg: dict, recipe: dict) -> ArmSpec:
    ext = a.get("extension") or {}
    lr = a["learning_rate"] if a["kind"] == "free_tensor" else recipe["learning_rate"]
    return ArmSpec(name=name, learning_rate=lr, steps=a["steps"], betas=tuple(recipe["adam_betas"]), weight_decay=0.0,
                   criterion_fraction=cfg["criterion_fraction"], report_every=cfg["report_every"],
                   save_steps=tuple(cfg["save_steps"]), latest_every_steps=cfg["latest_every_steps"],
                   extension_steps=ext.get("steps"), extension_lr_factor=ext.get("lr_factor", 0.1),
                   stop_at_criterion=bool(a.get("stop_at_criterion")), plateau=a.get("plateau"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/task23.yaml"))
    parser.add_argument("--device", choices=DEVICE_CHOICES, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stop-after", default=None, help="ARM:STEP -- simulate an interruption (resume test); exit 3")
    parser.add_argument("--deterministic", action="store_true",
                        help="torch.use_deterministic_algorithms(True): bit-reproducible updates where the backend supports "
                             "it (CPU yes; MPS raises for the interpolation backward). Off = the TASK-22 recipe.")
    args = parser.parse_args()
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    cfg = load_config(REPO_ROOT / args.config)
    recipe = resolve_recipe(cfg)
    run_dir, art = REPO_ROOT / cfg["run_root"] / args.run_name, REPO_ROOT / cfg["artifact_root"] / args.run_name
    if (art / "report.json").exists():
        raise SystemExit(f"{art / 'report.json'} exists: run {args.run_name} is finished and is never overwritten")
    device_request = args.device or cfg["device_request"]
    device = select_device(device_request, cfg["allow_cpu_fallback"])
    require_fp32_default()
    if torch.is_autocast_enabled():
        raise SystemExit("autocast must be disabled")
    device_info = device_metadata(device_request, device)
    environment = capture_environment_manifest(REPO_ROOT)
    print(f"TASK-23 {args.run_name} | device {json.dumps(device_info)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "task": "TASK-23", "run_name": args.run_name, "config_path": str(args.config), "config": cfg, "recipe": recipe,
        "uv_lock_sha256": file_sha256(REPO_ROOT / "uv.lock"), "starts": []}
    manifest["starts"].append({"utc": datetime.now(UTC).isoformat(), "device": device_info, "environment": environment, "argv": sys.argv,
                               "deterministic_algorithms": args.deterministic})
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, default=str))

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    stop_arm, stop_step = (args.stop_after.split(":") + [None])[:2] if args.stop_after else (None, None)

    pair = prepare_pair(cfg, recipe, device, run_dir / "pair.pt", art / "pair")
    meta = pair["meta"]
    T = pair["tensors"]
    feats, target, mask, positions = T["features"].to(device), T["target_mel"], T["mask"], T["positions"].to(device)
    t_mel, num_mels = target.shape[-1], target.shape[1]
    rf = MuQToMelTranslator(feats.shape[-1], recipe["hidden_channels"], num_mels).receptive_field_mel_frames
    boundary = boundary_frames(t_mel, (rf - 1) // 2, T["positions"], feats.shape[1])
    pair_meta = {"pair_sha256": pair["sha256"], "crop_id": meta["crop"]["crop_id"]}

    def decoder(seed: int) -> MuQToMelTranslator:
        seed_everything(seed, device)  # TASK-22 order: seed, then build on the CPU, then move
        return MuQToMelTranslator(feats.shape[-1], recipe["hidden_channels"], num_mels).to(device)

    def fwd(m):
        return m(feats, t_mel, positions)

    summaries, regs = {}, {}
    for name, a in cfg["arms"].items():
        if a.get("only_if_failed") and summaries.get(a["only_if_failed"], {}).get("criterion_met", True):
            print(f"[{name}] skipped: {a['only_if_failed']} met the criterion")
            continue
        spec = arm_spec(name, a, cfg, recipe)
        if a["kind"] == "free_tensor":
            module = FreeMel(tuple(target.shape), float(target.mean()), a["init_noise_std"], a["init_seed"]).to(device)
            forward = lambda m: m()
            spec.notes = {"init": f"target global mean {float(target.mean()):.4f} + {a['init_noise_std']} x N(0,1), seed {a['init_seed']}"}
        else:
            init = a["init"]
            if "same_as" in init:
                src = torch.load(run_dir / init["same_as"] / "checkpoints" / "milestone_000000.pt", map_location="cpu", weights_only=False)
                module = decoder(recipe["seed"])
                fresh_sha = state_dict_sha256(module)
                module.load_state_dict(src["model_state_dict"])
                if state_dict_sha256(module) != fresh_sha or src["state"]["init_sha256"] != fresh_sha:
                    raise SystemExit(f"STOP: {name}'s initial weights differ from {init['same_as']}'s")
                spec.notes = {"init": f"{init['same_as']}'s step-0 weights (sha {fresh_sha[:16]})"}
            else:
                seed = recipe["seed"] if init["seed"] == "recipe" else int(init["seed"])
                module = decoder(seed)
                spec.notes = {"init": f"fresh MuQToMelTranslator, seed {seed}" + (" (TASK-22's seed and order)" if seed == recipe["seed"] else "")}
            forward = fwd
            regs[f"{name}_audit"] = regularizer_audit(module, spec)
            if a.get("disable_regularizers"):
                regs[f"{name}_disabled"] = disable_regularizers(module, spec)
            n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            spec.notes |= {"trainable_parameters": n_trainable, "batch_size": 1,
                           "masked_l1_minus_task22_loss_at_init": float(masked_l1(fwd(module).detach(), target.to(device), mask.to(device))
                                                                        - (fwd(module).detach() - target.to(device)).abs().mean())}
        print(f"=== arm {name}: {spec.notes}")
        summ = train_arm(spec, module, forward, target, mask, boundary, run_dir / name, device, meta=pair_meta,
                         stop_after_step=int(stop_step) if stop_arm == name else None, should_stop=lambda: stop["flag"])
        if summ is None:
            return 3
        summaries[name] = summ
        print(f"[{name}] done: best eval L1 {summ['best']['eval_l1']:.6f} at step {summ['best']['step']} "
              f"({summ['best_relative_to_initial']:.3%} of initial); criterion met: {summ['criterion_met']}")
        if name == "A" and not summ["criterion_met"]:
            print("A missed the provisional criterion: the control failure is reported first (see the report); B/C still run")

    bigvgan = load_bigvgan(device)
    outputs = {k: arm_outputs(k, v, run_dir / k, art, pair, bigvgan, bigvgan.h, meta) for k, v in summaries.items()}
    b_c_identical = ("B" in summaries and "C" in summaries and summaries["B"]["final_sha256"] == summaries["C"]["final_sha256"])
    nondeterminism = None
    if "B" in summaries and "C" in summaries:
        hb, hc = ([json.loads(x) for x in (run_dir / k / "metrics.jsonl").read_text().splitlines()] for k in ("B", "C"))
        common = {e["step"]: e["eval/l1"] for e in hb}.keys() & {e["step"] for e in hc}
        eb, ec = {e["step"]: e["eval/l1"] for e in hb}, {e["step"]: e["eval/l1"] for e in hc}
        diffs = {s: abs(eb[s] - ec[s]) for s in sorted(common)}
        nondeterminism = {"deterministic_algorithms": args.deterministic, "b_c_final_weights_identical": b_c_identical,
                          "first_step_with_eval_l1_difference": next((s for s, d in diffs.items() if d > 0), None),
                          "max_abs_eval_l1_difference": max(diffs.values()) if diffs else None,
                          "note": ("B and C have identical initial weights and identical configurations (C disables nothing), so "
                                   "any B-C difference is run-to-run numerical nondeterminism (index_put accumulation in the "
                                   "frame-center interpolation backward), amplified by batch-1 Adam on an L1 loss")}
    report = {"task": "TASK-23", "run_name": args.run_name, "finished_utc": datetime.now(UTC).isoformat(), "device": device_info,
              "environment": environment, "config": cfg, "recipe_loop_facts": RECIPE_LOOP_FACTS,
              "pair_meta": {k: v for k, v in meta.items() if k != "selection_scores"} | {"selection_scores_top5": meta["selection_scores"][:5]},
              "pair_sha256": pair["sha256"], "summaries": summaries, "outputs": outputs,
              "regularizers": {"B_audit": regs.get("B_audit"), "C_disabled": regs.get("C_disabled", [])} | regs,
              "b_c_same_init": "B" in summaries and "C" in summaries and summaries["B"]["init_sha256"] == summaries["C"]["init_sha256"],
              "b_c_identical": b_c_identical, "nondeterminism": nondeterminism, "deterministic_algorithms": args.deterministic,
              "interpretation": interpret(summaries, nondeterminism), "listening": "pending"}
    write_report(art, report)
    atomic_write_text(art / "report.json", json.dumps(report, indent=2, default=str))
    print(f"report: {(art / 'TASK-23_RESULTS.md').relative_to(REPO_ROOT)} | {report['interpretation']['rule']}")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    code = main()
    print(f"elapsed {time.time() - t0:.0f} s")
    raise SystemExit(code)
