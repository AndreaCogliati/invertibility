"""Pure, testable helpers for the MuQ hidden-layer sweep (`experiment/muq-layer-sweep`).

Kept separate from the training/caching scripts so the deterministic-selection and
checkpoint-selection logic can be unit-tested without running any model.
"""

from __future__ import annotations

import dataclasses

LAYER_LABELS = ("earliest", "early_middle", "middle", "late_middle", "final")


def select_evenly_spaced_layers(num_transformer_layers: int, num_candidates: int = 5) -> list[int]:
    """Selects `num_candidates` approximately evenly spaced transformer-layer indices from
    `1` (output of the first transformer block) to `num_transformer_layers` (the final block,
    i.e. `last_hidden_state`) inclusive -- index `0` (the frontend/embedding output, before any
    transformer block) is deliberately excluded, since it is not itself a transformer hidden
    layer. Uses `linspace(1, num_transformer_layers, num_candidates)` rounded to the nearest
    integer, then deduplicated while preserving order (relevant only for very small
    `num_transformer_layers`). Pure function, deterministic: same inputs always give the same
    output, verified by a dedicated test rather than assumed.
    """
    if num_candidates < 2:
        raise ValueError("num_candidates must be >= 2 so the final layer and at least one earlier layer are both included")
    step = (num_transformer_layers - 1) / (num_candidates - 1)
    indices = [round(1 + i * step) for i in range(num_candidates)]
    # Deduplicate while preserving order, in case a very small num_transformer_layers collapses
    # two rounded positions onto the same index.
    seen: set[int] = set()
    deduped = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return deduped


def layer_cache_label(
    crop_label: str,
    layer_index: int,
    checksum_sha256: str,
    muq_model_id: str,
    muq_version: str,
    dtype: str,
) -> str:
    """Cache key for one (crop, MuQ-layer) pair. Includes every field this experiment's spec
    requires cache keys to carry: source-audio checksum, MuQ checkpoint identifier, MuQ package
    version, exact hidden-state index, and dtype (crop boundaries and preprocessing config are
    already embedded in `crop_label`, which is derived from the crop's split/track/start-time --
    see `gate4_common.CropSpec.label`)."""
    return f"{crop_label}__L{layer_index}__{checksum_sha256[:16]}__{muq_model_id}__v{muq_version}__{dtype}"


@dataclasses.dataclass(frozen=True)
class CheckpointRecord:
    step: int
    val_mse: float


def select_best_checkpoint(history: list[CheckpointRecord]) -> CheckpointRecord:
    """Returns the record with the lowest `val_mse` -- validation-MSE-only checkpoint selection,
    per this experiment's (and every prior MuQ-to-DAC experiment's) explicit requirement. Ties
    broken by earliest step, for determinism. Never looks at test performance or any other field,
    by construction: `CheckpointRecord` doesn't carry test metrics at all."""
    if not history:
        raise ValueError("history must be non-empty")
    return min(history, key=lambda r: (r.val_mse, r.step))
