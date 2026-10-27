"""TASK-22: first-conformer-block MuQ features (instead of last_hidden_state) -> BigVGAN mel.

MuQ's `hidden_states` tuple, as built by the installed transformers `Wav2Vec2ConformerEncoder`
(4.55.0; read from its source, then verified by forward hooks on every extraction batch):

    hidden_states[0]        input to conformer block 1 (Conv2dSubsampling frontend output; dropout is
                            identity in eval mode) -- NOT a block output
    hidden_states[k], 1<=k<=11   output of block k (`model.model.conformer.layers[k-1]`, after that
                            block's own final_layer_norm)
    hidden_states[12]       encoder LayerNorm(block 12 output) == last_hidden_state (the baseline)

So "the output of the first encoder block" is hidden_states[1] == layers[0]'s output. The last block
is not selectable here: its tuple entry carries the extra encoder LayerNorm, and it is the
`last_hidden_state` baseline path anyway.

Pure helpers (no model weights needed) so they can be unit-tested on a tiny real conformer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn


def resolve_muq_layer(spec: dict | None, num_blocks: int) -> dict | None:
    """`spec`: {"block": k} (1-based conformer block) or None (= last_hidden_state, the baseline).
    Returns the resolved identity used in configs, cache keys, manifests and W&B."""
    if spec is None:
        return None
    k = spec["block"]
    if not isinstance(k, int) or not 1 <= k < num_blocks:
        raise ValueError(f"muq_layer block must be an int in [1, {num_blocks - 1}] (block {num_blocks} is "
                         f"last_hidden_state, the baseline: use muq_layer: null); got {k!r}")
    return {"block": k, "hidden_state_index": k, "module": f"model.model.conformer.layers.{k - 1}",
            "name": f"conformer_block_{k}_output", "tag": f"L{k}", "num_blocks": num_blocks}


@torch.inference_mode()
def hidden_state_with_check(forward: Callable[[torch.Tensor], Sequence[torch.Tensor]], layers: nn.ModuleList,
                            x: torch.Tensor, block: int) -> tuple[torch.Tensor, dict]:
    """Run `forward(x) -> hidden_states` once with hooks on `layers`, and return hidden_states[block]
    (B, T, C) after checking -- every call, not assumed -- that:
      len(hidden_states) == len(layers) + 1;
      hidden_states[0] is exactly the input of layers[0] (so index 0 is not a block output);
      hidden_states[block] is exactly the output of layers[block - 1];
      hidden_states[block] differs from hidden_states[0]; all values finite.
    Raises RuntimeError on any failure."""
    got: dict[str, torch.Tensor] = {}
    h0 = layers[0].register_forward_pre_hook(lambda _m, args: got.__setitem__("in0", args[0].detach().clone()))
    h1 = layers[block - 1].register_forward_hook(lambda _m, _a, out: got.__setitem__("out", out[0].detach().clone()))
    try:
        hs = forward(x)
    finally:
        h0.remove()
        h1.remove()
    check = {"num_hidden_states": len(hs), "num_blocks": len(layers),
             "index0_is_block1_input": bool(torch.equal(hs[0], got["in0"])),
             "index_is_block_output": bool(torch.equal(hs[block], got["out"])),
             "differs_from_index0": not torch.equal(hs[block], hs[0]),
             "finite": bool(torch.isfinite(hs[block]).all())}
    if len(hs) != len(layers) + 1 or not all(v for k, v in check.items() if k not in ("num_hidden_states", "num_blocks")):
        raise RuntimeError(f"STOP: MuQ hidden-state indexing is not as verified for block {block}: {check}")
    return hs[block], check


def feature_cache_stem(layer: dict | None) -> str:
    """File-name stem of a feature cache: unchanged for last_hidden_state (so earlier caches are hit),
    layer-tagged otherwise. The digest that follows also covers the layer (it is in the key parts)."""
    return "task18_v1" if layer is None else f"task18_v1_{layer['tag']}"


def config_diff(base: dict, new: dict, ignore: Sequence[str] = ()) -> list[dict]:
    """Top-level keys whose values differ between two resolved configs (missing = None)."""
    return [{"key": k, "baseline": base.get(k), "task22": new.get(k)}
            for k in sorted(base.keys() | new.keys()) if k not in ignore and base.get(k) != new.get(k)]


def window_best(history: Sequence[dict], max_step: int, key: str = "val/correct/mel_l1") -> dict:
    """Lowest-`key` validation entry with step <= max_step (ties: earliest step)."""
    w = [e for e in history if e["step"] <= max_step]
    if not w:
        raise ValueError(f"no validation entries at or before step {max_step}")
    return min(w, key=lambda e: (e[key], e["step"]))


def checkpoint_steps(report: dict) -> dict[str, int]:
    """Evaluated-checkpoint label -> optimizer step, from a train_task18_heldout report."""
    out = {"best": report["best_checkpoint"]["step"], "final": report["final_step"]}
    for label, e in (report.get("extra_checkpoints") or {}).items():
        if e.get("available"):
            out[label] = e["step"]
    return out


def label_at_step(steps: dict[str, int], step: int) -> str | None:
    """An evaluated checkpoint label at exactly `step` (milestone labels preferred), or None."""
    hits = sorted((lab for lab, s in steps.items() if s == step), key=lambda lab: (lab in ("best", "final"), lab))
    return hits[0] if hits else None
