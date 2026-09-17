"""Tests for tasks/TASK-06.md's N=32 compute-controlled micro-ablation -- specifically
`rank_examples_by_metric`, added to `gate3_spectrogram_common.py` for this task's "identify
best/median/worst example" requirement.
"""

from __future__ import annotations

from muq_inversion.gate3_spectrogram_common import rank_examples_by_metric


def test_rank_examples_ascending_worst_first():
    per_example = {
        "a": {"waveform_correlation_to_oracle": 0.9},
        "b": {"waveform_correlation_to_oracle": 0.1},
        "c": {"waveform_correlation_to_oracle": 0.5},
    }
    ranking = rank_examples_by_metric(per_example)
    assert ranking == ["b", "c", "a"]
    worst, median, best = ranking[0], ranking[len(ranking) // 2], ranking[-1]
    assert worst == "b"
    assert median == "c"
    assert best == "a"


def test_rank_examples_custom_key():
    per_example = {
        "a": {"mse": 5.0, "waveform_correlation_to_oracle": 0.9},
        "b": {"mse": 1.0, "waveform_correlation_to_oracle": 0.1},
    }
    ranking = rank_examples_by_metric(per_example, key="mse")
    assert ranking == ["b", "a"]


def test_rank_examples_handles_32_examples_deterministically():
    per_example = {f"track_{i}": {"waveform_correlation_to_oracle": float(i)} for i in range(32)}
    ranking = rank_examples_by_metric(per_example)
    assert ranking[0] == "track_0"
    assert ranking[-1] == "track_31"
    assert ranking[16] == "track_16"  # median of 32 ascending, index len//2
    assert len(ranking) == 32
    assert len(set(ranking)) == 32
