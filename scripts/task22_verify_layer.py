"""TASK-22: verify, on the installed MuQ and real GTZAN crops, which hidden-state index is the output
of the FIRST conformer block, and record its geometry and alignment.

    uv run python scripts/task22_verify_layer.py [--device mps|xpu|cpu] [--out artifacts/task22/layer_verification]

Not assumed from documentation: forward hooks on every conformer block compare each `hidden_states`
entry with the actual block inputs/outputs of the same forward pass. Also recorded: shapes, frame rate,
token time centers (the conformer keeps the frontend's sequence, so token i has the same center at
every depth), mel-frame source positions, finiteness, determinism across two passes, the absence of
padding/masking (fixed-length crops, no attention mask), and how block-1 features differ from
last_hidden_state. Writes layer_verification.json.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import librosa
import numpy as np
import torch

from muq_inversion.audio import load_resample_validate
from muq_inversion.bigvgan_common import PEAK_NORMALIZE_TARGET, load_bigvgan
from muq_inversion.device import DEVICE_CHOICES
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.logging_utils import select_device
from muq_inversion.task22 import resolve_muq_layer
from muq_inversion.time_alignment import (
    bigvgan_mel_frame_centers,
    muq_token_centers,
    source_positions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
NUM_HOPS, HOP = 187, 256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--crops", type=Path, default=Path("data/manifests/task20_crops.csv"))
    parser.add_argument("--n-crops", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("artifacts/task22/layer_verification"))
    args = parser.parse_args()
    device = select_device(args.device, False)
    bigvgan = load_bigvgan(device)
    h = bigvgan.h
    muq = FrozenMuQEncoder(MUQ_MODEL_ID, device)
    inner = muq.model.model
    layers = inner.conformer.layers
    nb = len(layers)
    layer = resolve_muq_layer({"block": 1}, nb)

    # Real crops: one primary-val crop per track (deterministic), loaded exactly as build_features does.
    with open(REPO_ROOT / args.crops, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["eval_set"] == "val"]
    rows = [r for r in rows if r["crop_id"].endswith(":1")][: args.n_crops]
    n = NUM_HOPS * HOP
    waves = []
    for r in rows:
        track, _ = load_resample_validate(REPO_ROOT / r["path"], h.sampling_rate)
        track = librosa.util.normalize(track) * PEAK_NORMALIZE_TARGET
        s = int(r["crop_start_sample"])
        waves.append(torch.from_numpy(np.ascontiguousarray(track[s:s + n])).float())
    x = torch.stack(waves).to(device)

    # Hooks on every block: inputs and outputs of this very forward pass.
    ins, outs = {}, {}
    hooks = []
    for i, blk in enumerate(layers):
        hooks.append(blk.register_forward_pre_hook(lambda _m, a, i=i: ins.__setitem__(i, a[0].detach().clone())))
        hooks.append(blk.register_forward_hook(lambda _m, _a, o, i=i: outs.__setitem__(i, o[0].detach().clone())))
    with torch.inference_mode():
        out = muq.model(x, output_hidden_states=True)
        hs = out.hidden_states
        enc_ln_of_last = inner.conformer.layer_norm(outs[nb - 1])
    for hk in hooks:
        hk.remove()
    index_map = {}
    for k in range(len(hs)):
        matches = [f"input of block {i + 1}" for i in range(nb) if torch.equal(hs[k], ins[i])]
        matches += [f"output of block {i + 1}" for i in range(nb) if torch.equal(hs[k], outs[i])]
        if k == len(hs) - 1 and torch.equal(hs[k], enc_ln_of_last):
            matches.append(f"encoder layer_norm(output of block {nb})")
        if torch.equal(hs[k], out.last_hidden_state):
            matches.append("last_hidden_state")
        index_map[k] = matches

    # The hot path used by training (encoder.encode_block_output) on the same batch, twice.
    f1, check = muq.encode_block_output(x, layer["block"])
    f1b, _ = muq.encode_block_output(x, layer["block"])
    last = muq.encode(x)
    last_b = muq.encode(x)
    f1, f1b, last, last_b = (t.float().cpu() for t in (f1, f1b, last, last_b))
    t_muq = f1.shape[1]
    tok = muq_token_centers(inner, t_muq)
    mel_c = bigvgan_mel_frame_centers(h, NUM_HOPS)
    pos = source_positions(mel_c, tok)

    def stats(t: torch.Tensor) -> dict:
        return {"mean": float(t.mean()), "std": float(t.std()), "abs_max": float(t.abs().max()),
                "frame_l2_norm_mean": float(t.norm(dim=-1).mean()), "channel_std_mean": float(t.std(dim=(0, 1)).mean())}

    cos = torch.nn.functional.cosine_similarity(f1, last, dim=-1)
    report = {
        "muq_model_id": MUQ_MODEL_ID, "muq_version": importlib.metadata.version("muq"),
        "transformers_version": importlib.metadata.version("transformers"), "torch_version": torch.__version__,
        "device": str(device), "encoder_class": f"{type(inner.conformer).__module__}.{type(inner.conformer).__name__}",
        "num_conformer_blocks": nb, "num_hidden_states": len(hs),
        "hidden_state_index_map": index_map,
        "selected": layer, "hot_path_check": check,
        "extraction": {"call": "MuQ(x, output_hidden_states=True).hidden_states[1] under torch.inference_mode(); "
                       "model.eval(), all parameters requires_grad=False", "attention_mask": None,
                       "crop_num_samples": n, "crop_seconds": n / h.sampling_rate, "sample_rate": h.sampling_rate,
                       "padding_or_masking": "none: every crop is exactly 47,872 samples of real audio; MuQ is called "
                       "without attention_mask, so all tokens are valid", "feature_dtype": "float32",
                       "feature_normalization": "none (raw features into the translator's input Linear, as for last_hidden_state)"},
        "geometry": {"block1_shape": list(f1.shape), "last_hidden_state_shape": list(last.shape),
                     "frames_per_second": t_muq / (n / h.sampling_rate),
                     "same_shape_as_last_hidden_state": list(f1.shape) == list(last.shape),
                     "frontend_tokens_equal_block1_tokens": list(hs[0].shape) == list(hs[1].shape)},
        "alignment": {"token_centers_samples_first_last": [float(tok[0]), float(tok[-1])],
                      "token_hop_samples": float(np.diff(tok).mean()),
                      "note": "token centers come from the frontend (mel STFT + Conv2dSubsampling); conformer blocks "
                              "preserve sequence length and order (no striding/pooling), so they are identical for "
                              "hidden_states[1] and last_hidden_state",
                      "mel_source_positions_min_max": [float(pos.min()), float(pos.max())], "num_mel_frames": NUM_HOPS},
        "finite": {"block1": bool(torch.isfinite(f1).all()), "last": bool(torch.isfinite(last).all())},
        "determinism": {"block1_bit_exact_two_passes": bool(torch.equal(f1, f1b)), "block1_max_abs_diff": float((f1 - f1b).abs().max()),
                        "last_bit_exact_two_passes": bool(torch.equal(last, last_b)), "last_max_abs_diff": float((last - last_b).abs().max())},
        "block1_vs_last": {"block1": stats(f1), "last_hidden_state": stats(last),
                           "per_frame_cosine_mean": float(cos.mean()), "per_frame_cosine_min_max": [float(cos.min()), float(cos.max())]},
        "crops": [r["crop_id"] for r in rows],
    }
    ok = (index_map[0] == ["input of block 1"] and "output of block 1" in index_map[1]
          and index_map[len(hs) - 1] == [f"encoder layer_norm(output of block {nb})", "last_hidden_state"]
          and report["geometry"]["same_shape_as_last_hidden_state"] and all(report["finite"].values()))
    report["verdict"] = "PASS" if ok else "FAIL"
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "layer_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("num_hidden_states", "hidden_state_index_map", "geometry", "determinism",
                                             "block1_vs_last", "verdict")}, indent=1))
    print(f"wrote {(out / 'layer_verification.json').relative_to(REPO_ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
