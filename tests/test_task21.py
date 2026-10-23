"""TASK-21: schedules, runtime budget, retention, and 'only duration/stopping/logging changed'."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from muq_inversion.task21 import RuntimeBudget, due, recovery_to_delete

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_task18_heldout import load_config

T20 = load_config(REPO_ROOT / "configs/task20.yaml")
T21 = load_config(REPO_ROOT / "configs/task21.yaml")
ALLOWED = {  # duration, stopping, logging, checkpoint management, continuation, naming
    "task", "init_from_checkpoint", "early_stopping", "steps", "runtime_budget_hours", "train_only", "eval_schedule",
    "log_schedule", "quick_eval_sets", "checkpoint_policy", "arms", "run_name_prefix", "run_root", "artifact_root",
    "results_name", "wandb_group", "wandb_tags",
}


def test_only_duration_stopping_logging_and_checkpointing_changed():
    changed = {k for k in T20.keys() | T21.keys() if T20.get(k) != T21.get(k)}
    assert changed <= ALLOWED, changed - ALLOWED
    assert T21["arms"]["N512ext"] == T20["arms"]["N512"]  # same tracks policy and constant LR
    for k in ("track_manifest", "crop_manifest", "eval_sets", "learning_rate", "adam_betas", "batch_size", "seed",
              "hidden_channels", "temporal_alignment", "decode_clamp_to_mel_floor", "loss"):
        assert T21[k] == T20[k], k


def test_continuation_from_task20_last_not_best_and_no_early_stopping():
    init = T21["init_from_checkpoint"]
    assert init["latest"].endswith("task20_N512_v1/checkpoints/latest.pt")
    assert init["best"].endswith("task20_N512_v1/checkpoints/best.pt")
    assert T21["early_stopping"] is None and T21["steps"] is None and T21["runtime_budget_hours"] == 48
    assert not {"test", "test2", "val2"} & set(T21["quick_eval_sets"]) and "val" in T21["quick_eval_sets"]


def test_eval_schedule():
    sched = T21["eval_schedule"]
    assert [s for s in range(1, 20001) if due(s, sched)][:3] == [250, 500, 750]
    assert due(20000, sched) and not due(20250, sched) and due(21000, sched) and due(1_000_000, sched)
    assert sum(due(s, sched) for s in range(4251, 25001)) == (20000 - 4250) // 250 + 5


def test_runtime_budget_counts_previous_segments():
    b = RuntimeBudget(1.0, used_before=3000.0, segment_start=100.0)
    assert not b.exhausted(now=600.0) and b.exhausted(now=700.0)
    assert RuntimeBudget(None, 0.0, 0.0).exhausted(now=1e12) is False


def test_recovery_retention():
    files = [Path(f"recovery_{s:09d}.pt") for s in (5000, 1000, 3000, 2000)]
    assert recovery_to_delete(files, keep=3) == [Path("recovery_000001000.pt")]
    assert recovery_to_delete(files[:2], keep=3) == []


@pytest.mark.skipif(not (REPO_ROOT / "runs/task20/task20_N512_v1/checkpoints/latest.pt").exists(),
                    reason="TASK-20 checkpoints not present locally")
def test_task20_last_checkpoint_is_complete_for_exact_continuation():
    from muq_inversion.device import load_portable

    ck = load_portable(REPO_ROOT / "runs/task20/task20_N512_v1/checkpoints/latest.pt")
    st = ck["train_state"]
    assert st["step"] == 4250 and st["best"]["step"] == 2250
    assert {int(float(s["step"])) for s in ck["optimizer_state_dict"]["state"].values()} == {4250}
    assert {"rng_state", "queue", "track_exposure", "crop_exposure", "num_batches"} <= set(ck["sampler_state"])
    assert {"torch_cpu_rng", "python_rng", "numpy_rng"} <= set(ck)
    assert ck["sampler_state"]["num_batches"] == 4250
