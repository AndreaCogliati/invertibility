"""TASK-26 report: measure the implemented pipeline's geometry on one real crop (read-only).

Loads the frozen MuQ checkpoint and the pinned BigVGAN code/config, takes the TASK-23 anchor pair
(runs/task23/task23_v1/pair.pt), and prints every shape, rate, center convention and parameter count
the report cites. Writes `geometry.json` next to this script. Does not train or modify anything.

    uv run python reports/task26_pipeline_report/scripts/inspect_geometry.py --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from muq_inversion.bigvgan_common import bigvgan_log_mel_floor, load_bigvgan, official_mel
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.models.muq_mel_translator import MuQToMelTranslator
from muq_inversion.task25 import build_model, forward_macs, source_receptive_field
from muq_inversion.time_alignment import bigvgan_mel_frame_centers, muq_token_centers, source_positions

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "geometry.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    g: dict = {}

    pair = torch.load(REPO / "runs/task23/task23_v1/pair.pt", map_location="cpu", weights_only=False)
    g["pair_keys"] = sorted(pair.keys())
    tens = pair["tensors"]
    g["pair_tensors"] = {k: [list(v.shape), str(v.dtype)] for k, v in tens.items()}
    wave = tens["waveform"].reshape(1, -1).float()
    tmel = tens["target_mel"].float()
    feat = tens["features"].float()
    g["pair_meta"] = pair["meta"]
    g["cached_positions_first3_last3"] = [*tens["positions"][:3].tolist(), *tens["positions"][-3:].tolist()]
    g["mask_shape_sum"] = [list(tens["mask"].shape), float(tens["mask"].sum())]

    bigvgan = load_bigvgan(dev)
    h = bigvgan.h
    g["bigvgan"] = {k: h[k] for k in ("sampling_rate", "num_mels", "n_fft", "win_size", "hop_size", "fmin", "fmax",
                                      "upsample_rates", "normalize_volume")}
    g["bigvgan"]["upsample_product"] = int(np.prod(h.upsample_rates))
    g["bigvgan"]["params"] = sum(p.numel() for p in bigvgan.parameters())
    g["bigvgan"]["requires_grad_any"] = any(p.requires_grad for p in bigvgan.parameters())
    g["bigvgan"]["log_mel_floor"] = bigvgan_log_mel_floor()
    g["bigvgan"]["reflect_pad"] = (h.n_fft - h.hop_size) // 2

    g["crop"] = {"num_samples": wave.shape[-1], "seconds": wave.shape[-1] / h.sampling_rate,
                 "peak": float(wave.abs().max()), "rms": float(wave.pow(2).mean().sqrt())}
    mel = official_mel(wave, h)
    g["target_mel"] = {"shape": list(mel.shape), "dtype": str(mel.dtype), "min": float(mel.min()),
                       "max": float(mel.max()), "mean": float(mel.mean()),
                       "max_abs_diff_vs_cached": float((mel - tmel.reshape(mel.shape)).abs().max()),
                       "frames_per_second": h.sampling_rate / h.hop_size,
                       "bins_at_floor": int((mel <= bigvgan_log_mel_floor() + 1e-6).sum())}
    with torch.inference_mode():
        audio = bigvgan(mel.clamp(min=bigvgan_log_mel_floor()).to(dev)).cpu()
    g["bigvgan_output"] = {"shape": list(audio.shape), "dtype": str(audio.dtype)}

    muq = FrozenMuQEncoder("OpenMuQ/MuQ-large-msd-iter", dev)
    m = muq.model.model
    spec = m.preprocessor_melspec_2048.mel_stft
    g["muq"] = {
        "model_id": "OpenMuQ/MuQ-large-msd-iter",
        "params": sum(p.numel() for p in muq.model.parameters()),
        "training_mode": muq.model.training,
        "requires_grad_any": any(p.requires_grad for p in muq.model.parameters()),
        "frontend": {"sample_rate": spec.sample_rate, "n_fft": spec.spectrogram.n_fft,
                     "win_length": spec.spectrogram.win_length, "hop_length": spec.spectrogram.hop_length,
                     "center": spec.spectrogram.center, "pad": spec.spectrogram.pad,
                     "power": spec.spectrogram.power, "n_mels": spec.n_mels,
                     "f_min": spec.mel_scale.f_min, "f_max": spec.mel_scale.f_max,
                     "mel_scale": spec.mel_scale.mel_scale, "norm": spec.mel_scale.norm,
                     "is_db": m.preprocessor_melspec_2048.is_db,
                     "db_top_db": getattr(getattr(m.preprocessor_melspec_2048, "amplitude_to_db", None), "top_db", None),
                     "stat_keys": sorted(m.stat.keys()) if m.stat else [],
                     "stat_values": {k: float(v) for k, v in (m.stat or {}).items()}},
        "n_fold": m.n_fold, "label_rate": m.label_rate,
        "conv": [{"conv1_stride": list(b.conv1.stride), "conv1_kernel": list(b.conv1.kernel_size),
                  "conv1_padding": list(b.conv1.padding), "out_channels": b.conv1.out_channels,
                  "shortcut": bool(getattr(b, "diff", False))} for b in m.conv.conv],
        "conv_linear": [m.conv.linear.in_features, m.conv.linear.out_features],
        "conformer_layers": len(m.conformer.layers),
        "has_rvq_attr": hasattr(m, "rvq"),
        "quantizer_attrs": [k for k in dir(m) if k.startswith("quantizer_")],
        "is_flash": muq.model.config.is_flash,
        "w2v2_config": {k: v for k, v in dict(muq.model.config.w2v2_config).items()
                        if isinstance(v, (int, float, str, bool)) or v is None},
    }
    with torch.inference_mode():
        x = wave.to(dev)
        fe = m.preprocessing(x, features=["melspec_2048"])["melspec_2048"]
        g["muq"]["frontend_out_shape"] = list(fe.shape)
        fe_n = m.normalize({"melspec_2048": fe.clone()})["melspec_2048"]
        g["muq"]["frontend_normalized_mean_std"] = [float(fe_n.mean()), float(fe_n.std())]
        cv = m.conv(fe_n)
        g["muq"]["conv_out_shape"] = list(cv.shape)
        hs = muq.model(x, output_hidden_states=True).hidden_states
        g["muq"]["num_hidden_states"] = len(hs)
        g["muq"]["hidden_state_shapes"] = {i: list(t.shape) for i, t in enumerate(hs) if i in (0, 1, 12)}
        g["muq"]["hidden0_equals_conv_out"] = bool(torch.allclose(hs[0], cv, atol=1e-5))
        b1, chk = muq.encode_block_output(x, 1)
        g["muq"]["block1_check"] = chk
        g["muq"]["block1_vs_cached_max_abs"] = float((b1.cpu() - feat.reshape(b1.shape)).abs().max())
        g["muq"]["block1_vs_cached_rel_rms"] = float((b1.cpu() - feat.reshape(b1.shape)).pow(2).mean().sqrt()
                                                     / feat.pow(2).mean().sqrt())
        g["muq"]["block1_std"] = float(b1.std())
        g["muq"]["last_std"] = float(hs[12].std())
        g["muq"]["dtype"] = str(b1.dtype)
        g["muq"]["finite"] = bool(torch.isfinite(b1).all())

    t_muq, t_mel = hs[1].shape[1], mel.shape[-1]
    tc = muq_token_centers(m, t_muq)
    mc = bigvgan_mel_frame_centers(h, t_mel)
    pos = source_positions(mc, tc)
    g["alignment"] = {
        "muq_token_centers_first3_last": [*tc[:3].tolist(), float(tc[-1])],
        "muq_token_hop_samples": float(tc[1] - tc[0]),
        "muq_frames_per_second_nominal": h.sampling_rate / float(tc[1] - tc[0]),
        "muq_frames_per_second_crop": t_muq / (wave.shape[-1] / h.sampling_rate),
        "mel_frame_centers_first3_last": [*mc[:3].tolist(), float(mc[-1])],
        "positions_first3_last3": [*pos[:3].tolist(), *pos[-3:].tolist()],
        "num_positions_clamped_high": int((pos >= t_muq - 1).sum()),
        "num_positions_clamped_low": int((pos <= 0).sum()),
        "mel_per_muq": float(tc[1] - tc[0]) / h.hop_size,
        "last_token_center_s": float(tc[-1]) / h.sampling_rate,
        "last_mel_center_s": float(mc[-1]) / h.sampling_rate,
    }

    tr = MuQToMelTranslator(1024, 256, 100)
    g["translator"] = {"params": sum(p.numel() for p in tr.parameters()),
                       "receptive_field_mel_frames": tr.receptive_field_mel_frames,
                       "receptive_field_seconds": tr.receptive_field_mel_frames * h.hop_size / h.sampling_rate}
    for arm, mdl in (("current", build_model("current", 1024, 256, 100)),
                     ("wide_x2", build_model("current", 1024, 256, 100, width_multiplier=2)),
                     ("learned_resampler", build_model("learned_resampler", 1024, 256, 100))):
        g["translator"][arm] = {"params": sum(p.numel() for p in mdl.parameters()),
                                "macs": forward_macs(mdl, t_muq, t_mel)["total"],
                                "source_rf": source_receptive_field(mdl, pos, t_muq)}

    OUT.write_text(json.dumps(g, indent=1, default=str))
    print(json.dumps(g, indent=1, default=str))


if __name__ == "__main__":
    main()
