"""Tests for tasks/TASK-05.md's scaling-curve helpers added to
`src/muq_inversion/gate3_spectrogram_common.py`: `aggregate_metrics_across_examples` (mean/
median/min/max across N examples) and `load_examples_from_manifest` (the generalized loader
behind TASK-04's `load_examples`).
"""

from __future__ import annotations

import pytest
import torch

from muq_inversion.gate3_spectrogram_common import (
    DENSE_CACHE_PATH,
    aggregate_metrics_across_examples,
    load_examples_from_manifest,
)

requires_dense_cache = pytest.mark.skipif(
    not DENSE_CACHE_PATH.exists(), reason="data/cache/dense/mel_dac_cache.pt not present -- run scripts/precompute_dense_cache.py"
)


def test_aggregate_metrics_computes_mean_median_min_max():
    per_example = {
        "a": {"mse": 1.0, "cosine": 0.9},
        "b": {"mse": 3.0, "cosine": 0.5},
        "c": {"mse": 2.0, "cosine": 0.7},
    }
    agg = aggregate_metrics_across_examples(per_example)
    assert agg["mse"] == {"mean": 2.0, "median": 2.0, "min": 1.0, "max": 3.0}
    assert agg["cosine"]["mean"] == pytest.approx(0.7)
    assert agg["cosine"]["min"] == 0.5
    assert agg["cosine"]["max"] == 0.9


def test_aggregate_metrics_single_example():
    per_example = {"only": {"mse": 5.0}}
    agg = aggregate_metrics_across_examples(per_example)
    assert agg["mse"] == {"mean": 5.0, "median": 5.0, "min": 5.0, "max": 5.0}


def test_aggregate_metrics_rejects_empty_input():
    with pytest.raises(ValueError):
        aggregate_metrics_across_examples({})


def test_aggregate_metrics_rejects_mismatched_keys():
    per_example = {
        "a": {"mse": 1.0, "cosine": 0.9},
        "b": {"mse": 3.0},  # missing "cosine"
    }
    with pytest.raises(ValueError):
        aggregate_metrics_across_examples(per_example)


def test_aggregate_metrics_does_not_silently_drop_a_worst_example():
    """A single bad example must move min/max even when N is large -- the exact 'worst-case
    example may become particularly informative' property tasks/TASK-05.md calls out."""
    per_example = {f"track_{i}": {"mse": 0.01} for i in range(50)}
    per_example["outlier"] = {"mse": 100.0}
    agg = aggregate_metrics_across_examples(per_example)
    assert agg["mse"]["max"] == 100.0
    assert agg["mse"]["median"] == pytest.approx(0.01)  # median robust to the one outlier


@requires_dense_cache
def test_load_examples_from_manifest_generalizes_to_arbitrary_manifest():
    from pathlib import Path

    manifest_path = Path("data/manifests/gate3_scale_8.csv")
    if not manifest_path.exists():
        pytest.skip("gate3_scale_8.csv not generated -- run scripts/build_gate3_scale_manifests.py")
    examples = load_examples_from_manifest(manifest_path, torch.device("cpu"))
    assert len(examples) == 8
    for ex in examples.values():
        assert ex.mel.shape[1] == ex.dac.shape[1] == 150
