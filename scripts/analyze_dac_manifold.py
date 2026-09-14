"""Offline DAC-space validity analysis for notes/EXPERIMENT_SYNTHESIS.md.

Read-only diagnostic: loads the already-trained `more_data` checkpoint (best-by-validation-MSE,
step 50000) and the already-cached MuQ/DAC latents, runs inference-only forward passes (NO
training, NO gradient updates) to compare predicted vs. oracle DAC latents on held-out (val)
crops. Answers: are held-out predictions merely inaccurate, or also off the manifold of vectors
DAC's own residual vector quantizer would actually produce?

Does not modify any existing checkpoint, cache, or report.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from transformers import DacModel

from muq_inversion.logging_utils import select_device
from muq_inversion.models.temporal_conv import TemporalConvNet

REPO_ROOT_MANIFEST = "data/manifests/more_data_crops.csv"
CACHE_PATH = "data/cache/more_data/muq_dac_cache.pt"
CHECKPOINT_PATH = "runs/more_data/checkpoints/best.pt"
N_VAL_CROPS_TO_ANALYZE = 40  # a subsample of the 188 val crops -- cheap, sufficient for stats


def main() -> None:
    device = select_device("auto", False)
    print(f"device: {device}")

    cache = torch.load(CACHE_PATH, map_location=device, weights_only=False)
    frame_ratio = cache["frame_ratio"]
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = TemporalConvNet(channels=1024).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint step {ckpt['step']}, val_mse={ckpt['val_mse']:.4f}")

    dac = DacModel.from_pretrained("descript/dac_24khz").to(device)
    dac.eval()

    import csv

    with open(REPO_ROOT_MANIFEST, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    val_labels = []
    for r in rows:
        label = f"val_{r['track_id']}_t{float(r['crop_start_seconds']):.1f}s"
        if label in cache["crops"]:
            val_labels.append(label)
        if len(val_labels) >= N_VAL_CROPS_TO_ANALYZE:
            break

    print(f"analyzing {len(val_labels)} val crops")

    predicted_all, oracle_all = [], []
    requant_all = []
    codebook_agreement_per_stage = []
    with torch.no_grad():
        for label in val_labels:
            muq_feat = cache["crops"][label]["muq"].to(device)
            dac_target = cache["crops"][label]["dac"].to(device)  # (1, T, 1024)
            repeated = muq_feat.repeat_interleave(frame_ratio, dim=1)
            pred = model(repeated)  # (1, T, 1024)

            predicted_all.append(pred.squeeze(0).cpu().numpy())
            oracle_all.append(dac_target.squeeze(0).cpu().numpy())

            # Re-quantize the prediction through DAC's own frozen RVQ (structurally valid: the
            # quantizer's expected input width equals quantized_representation's width, 1024).
            pred_bct = pred.transpose(1, 2)  # (1, 1024, T)
            target_bct = dac_target.transpose(1, 2)
            requant, pred_codes, _, _, _ = dac.quantizer(pred_bct, None)
            _, oracle_codes, _, _, _ = dac.quantizer(target_bct, None)
            requant_all.append(requant.transpose(1, 2).squeeze(0).cpu().numpy())

            agreement = (pred_codes == oracle_codes).float().mean(dim=(0, 2))  # per-stage, over time
            codebook_agreement_per_stage.append(agreement.cpu().numpy())

    predicted = np.concatenate(predicted_all, axis=0)  # (total_frames, 1024)
    oracle = np.concatenate(oracle_all, axis=0)
    requant = np.concatenate(requant_all, axis=0)

    def frame_diff_stats(x: np.ndarray) -> dict:
        diffs = np.diff(x, axis=0)
        return {"mean_abs_frame_diff": float(np.mean(np.abs(diffs))), "std_frame_diff": float(np.std(diffs))}

    def autocorr_lag1(x: np.ndarray) -> float:
        x_centered = x - x.mean(axis=0, keepdims=True)
        num = np.sum(x_centered[:-1] * x_centered[1:])
        den = np.sum(x_centered[:-1] ** 2) + 1e-9
        return float(num / den)

    pred_norms = np.linalg.norm(predicted, axis=1)
    oracle_norms = np.linalg.norm(oracle, axis=1)
    requant_dist = np.linalg.norm(predicted - requant, axis=1)  # distance to nearest valid RVQ point
    codebook_agreement = np.stack(codebook_agreement_per_stage, axis=0).mean(axis=0)  # (32,)

    report = {
        "num_crops_analyzed": len(val_labels),
        "num_frames_analyzed": int(predicted.shape[0]),
        "predicted": {
            "mean_of_channel_means": float(predicted.mean()),
            "mean_of_channel_vars": float(predicted.var(axis=0).mean()),
            "vector_norm_mean": float(pred_norms.mean()),
            "vector_norm_std": float(pred_norms.std()),
            **frame_diff_stats(predicted),
            "lag1_autocorr": autocorr_lag1(predicted),
        },
        "oracle": {
            "mean_of_channel_means": float(oracle.mean()),
            "mean_of_channel_vars": float(oracle.var(axis=0).mean()),
            "vector_norm_mean": float(oracle_norms.mean()),
            "vector_norm_std": float(oracle_norms.std()),
            **frame_diff_stats(oracle),
            "lag1_autocorr": autocorr_lag1(oracle),
        },
        "requantization_analysis": {
            "distance_to_nearest_valid_rvq_point_mean": float(requant_dist.mean()),
            "distance_to_nearest_valid_rvq_point_std": float(requant_dist.std()),
            "relative_to_prediction_norm": float(requant_dist.mean() / pred_norms.mean()),
            "codebook_index_agreement_pred_vs_oracle_per_stage": codebook_agreement.tolist(),
            "codebook_index_agreement_mean_over_32_stages": float(codebook_agreement.mean()),
            "codebook_index_agreement_stage0_(coarsest)": float(codebook_agreement[0]),
        },
    }
    print(json.dumps(report, indent=2))

    out_path = "artifacts/synthesis/dac_manifold_analysis.json"
    import os

    os.makedirs("artifacts/synthesis", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
