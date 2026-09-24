"""Regenerates the Gate 5 listening set -- worst/median/best held-out crop per split by
oracle-relative waveform recovery -- from a finished run's checkpoint and report.json, without
retraining.

Written because TASK-14's full run (BlueHive job 1956835) saved its listening WAVs with the
original first-N-in-sort-order selection, which yielded six blues crops from three tracks. Uses
the same `select_listening_labels` and `write_listening_artifacts` as
`scripts/train_gate5_full_gtzan_wandb.py`, so the output matches what the fixed training script
would have written.

    uv run python scripts/export_gate5_listening.py \\
        --config configs/gate5_full_gtzan.yaml \\
        --checkpoint runs/task14_gate5_full_gtzan/checkpoints/best.pt \\
        --report bluehive/artifacts/task14_gate5_full_gtzan/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from muq_inversion.encodec_encoder import FrozenEncodecEncoder
from muq_inversion.encoder import FrozenMuQEncoder
from muq_inversion.gate4_encodec_common import load_gate4_examples
from muq_inversion.gate5_common import (
    load_gate5_config,
    select_listening_labels,
    write_listening_artifacts,
)
from muq_inversion.logging_utils import select_device
from muq_inversion.models.spectrogram_translator import SpectrogramToDacTranslator

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_METRIC = "oracle_relative_waveform_recovery"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, help="default: artifacts/<artifact_dir_name>/listening_representative")
    args = parser.parse_args()

    config = load_gate5_config(args.config)
    out_dir = args.out_dir or REPO_ROOT / "artifacts" / config.artifact_dir_name / "listening_representative"
    report = json.loads(args.report.read_text())
    correct = report["control_report"]["correct"]["per_example"]

    selection = {
        split: select_listening_labels(correct, sorted(lbl for lbl in correct if lbl.startswith(f"{split}_")), PRIMARY_METRIC)
        for split in ("val", "test")
    }
    for split, roles in selection.items():
        print(f"{split}: " + ", ".join(f"{role}={lbl} ({correct[lbl][PRIMARY_METRIC]:.4f})" for role, lbl in roles.items()))

    device = select_device(requested=config.device_request, allow_cpu_fallback=config.allow_cpu_fallback)
    muq = FrozenMuQEncoder(config.muq_model_id, device)
    encodec = FrozenEncodecEncoder(config.encodec_model_id, device, bandwidth=config.encodec_bandwidth_kbps)
    # All held-out crops, not just the six selected: shuffled_across draws its source from this pool.
    heldout_examples, frame_ratio = load_gate4_examples(
        REPO_ROOT / config.heldout_manifest, "gate5", muq, encodec, config.sample_rate, device
    )

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SpectrogramToDacTranslator(input_channels=1024, dac_channels=128).to(device)
    model.load_state_dict(state["model_state_dict"])
    print(f"checkpoint step {state.get('step')}")

    labels = [lbl for roles in selection.values() for lbl in roles.values()]
    write_listening_artifacts(labels, heldout_examples, model, encodec, frame_ratio, config.seed, out_dir, config.sample_rate)
    (out_dir / "selection.json").write_text(json.dumps({"metric": PRIMARY_METRIC, "checkpoint": str(args.checkpoint), **selection}, indent=2))
    print(f"wrote {len(labels) * 6} WAVs + selection.json to {out_dir}")


if __name__ == "__main__":
    main()
