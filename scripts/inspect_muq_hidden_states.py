"""Gate-0-style diagnostic: measure MuQ's `output_hidden_states=True` structure directly rather
than assuming it, per CLAUDE.md's "do not assume the frame rate or channel dimension" rule
applied to layer indexing as well.

Records:
- total number of returned hidden states;
- whether index 0 is the frontend/embedding output or the first transformer block (checked
  against the model's actual submodule structure, not inferred from shape alone);
- tensor shape, frame rate, dtype, and channel count for every hidden state;
- the exact 5 candidate layer indices this experiment uses, and their semantic labels.

Writes `artifacts/muq_layer_sweep/hidden_states_inspection.json` and prints a summary.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import torch
from muq import MuQ

from muq_inversion.layer_sweep_common import LAYER_LABELS, select_evenly_spaced_layers
from muq_inversion.logging_utils import DeviceSelectionError, select_device

REPO_ROOT = Path(__file__).resolve().parents[1]
MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
SAMPLE_RATE = 24000
CROP_SECONDS = 2.0


def main() -> None:
    try:
        device = select_device(requested="auto", allow_cpu_fallback=False)
    except DeviceSelectionError as exc:
        raise SystemExit(f"Device selection failed: {exc}") from exc
    print(f"device: {device}")

    muq_version = importlib.metadata.version("muq")
    model = MuQ.from_pretrained(MUQ_MODEL_ID)
    model.to(device)
    model.eval()

    # Measure the actual submodule structure -- do not infer "index 0 is the embedding output"
    # from shape alone.
    num_transformer_layers = len(model.model.conformer.layers)
    print(f"MuQ conformer transformer layers (measured from model.model.conformer.layers): {num_transformer_layers}")

    waveform = torch.randn(1, int(CROP_SECONDS * SAMPLE_RATE), device=device)
    with torch.inference_mode():
        output = model(waveform, output_hidden_states=True)
    hidden_states = output.hidden_states
    num_hidden_states = len(hidden_states)
    print(f"output_hidden_states=True returns {num_hidden_states} tensors")

    index0_is_embedding = num_hidden_states == num_transformer_layers + 1
    assert index0_is_embedding, (
        f"STOP CONDITION: expected num_hidden_states ({num_hidden_states}) == "
        f"num_transformer_layers + 1 ({num_transformer_layers + 1}); indexing convention "
        f"assumption is wrong, do not proceed without re-deriving it"
    )
    final_matches_last_hidden_state = torch.equal(hidden_states[-1], output.last_hidden_state)
    print(f"index 0 is the frontend/embedding output (measured): {index0_is_embedding}")
    print(f"hidden_states[-1] == last_hidden_state (measured): {final_matches_last_hidden_state}")
    assert final_matches_last_hidden_state, "STOP CONDITION: final hidden state does not match last_hidden_state"

    layer_shapes = []
    for i, h in enumerate(hidden_states):
        layer_shapes.append(
            {
                "index": i,
                "shape": list(h.shape),
                "channels": h.shape[-1],
                "num_frames": h.shape[1],
                "frames_per_second": h.shape[1] / CROP_SECONDS,
                "dtype": str(h.dtype),
                "is_finite": bool(torch.isfinite(h).all().item()),
            }
        )
        print(f"  index {i}: shape={tuple(h.shape)} dtype={h.dtype} finite={layer_shapes[-1]['is_finite']}")

    channels = {ls["channels"] for ls in layer_shapes}
    frame_rates = {ls["frames_per_second"] for ls in layer_shapes}
    print(f"distinct channel counts across all hidden states: {channels}")
    print(f"distinct frame rates across all hidden states: {frame_rates}")
    adapter_needed = len(channels) > 1 or len(frame_rates) > 1
    print(f"shape adapter needed: {adapter_needed}")

    candidate_indices = select_evenly_spaced_layers(num_transformer_layers, num_candidates=5)
    assert candidate_indices[-1] == num_transformer_layers, "final candidate must be the last transformer layer"
    candidates = [
        {"label": label, "index": idx, **{k: v for k, v in layer_shapes[idx].items() if k != "index"}}
        for label, idx in zip(LAYER_LABELS, candidate_indices, strict=True)
    ]
    print("\nSelected 5 candidate layers:")
    for c in candidates:
        print(f"  {c['label']:>13s} = index {c['index']:2d}  shape={tuple(c['shape'])} finite={c['is_finite']}")

    report = {
        "muq_model_id": MUQ_MODEL_ID,
        "muq_version": muq_version,
        "device": str(device),
        "num_transformer_layers": num_transformer_layers,
        "num_hidden_states": num_hidden_states,
        "index0_is_embedding_output": index0_is_embedding,
        "final_hidden_state_matches_last_hidden_state": final_matches_last_hidden_state,
        "shape_adapter_needed": adapter_needed,
        "all_layer_shapes": layer_shapes,
        "candidate_layers": candidates,
    }
    out_path = REPO_ROOT / "artifacts/muq_layer_sweep/hidden_states_inspection.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
