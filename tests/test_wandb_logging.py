"""Tests the network-free parts of `wandb_logging.py` only (run-name generation, metadata
building, checkpoint-table selection) -- no live W&B connection or `wandb.init()` call, so these
run in any environment regardless of network access or API-key configuration.
"""

from __future__ import annotations

import datetime

from muq_inversion.wandb_logging import (
    build_run_metadata,
    checkpoint_table_rows,
    make_run_name,
)


def test_make_run_name_format():
    name = make_run_name(now=datetime.datetime(2026, 9, 14, 10, 30, 0, tzinfo=datetime.UTC))
    assert name == "spec_dense_gtzan_20260914T103000Z"


def test_make_run_name_custom_prefix():
    name = make_run_name(prefix="foo", now=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC))
    assert name.startswith("foo_")


def test_make_run_name_is_unique_across_seconds():
    a = make_run_name(now=datetime.datetime(2026, 9, 14, 10, 30, 0, tzinfo=datetime.UTC))
    b = make_run_name(now=datetime.datetime(2026, 9, 14, 10, 30, 1, tzinfo=datetime.UTC))
    assert a != b


def test_build_run_metadata_pulls_expected_fields():
    env = {
        "git_commit": "abc123",
        "git_dirty": False,
        "hostname": "node042",
        "machine": "x86_64",
        "platform": "Linux-5.15",
        "cuda_available": True,
        "cuda_device_name": "NVIDIA A100",
        "mps_available": False,
        "slurm_job_id": "998877",
    }
    meta = build_run_metadata(env, job_id=None)
    assert meta["git_commit"] == "abc123"
    assert meta["hostname"] == "node042"
    assert meta["machine_type"] == "x86_64"
    assert meta["cuda_device_name"] == "NVIDIA A100"
    assert meta["slurm_job_id"] == "998877"


def test_build_run_metadata_job_id_arg_overrides_environment():
    env = {"slurm_job_id": "from_env"}
    meta = build_run_metadata(env, job_id="from_arg")
    assert meta["slurm_job_id"] == "from_arg"


def test_build_run_metadata_falls_back_to_platform_node_for_hostname():
    meta = build_run_metadata({}, job_id=None)
    assert meta["hostname"]  # non-empty, from platform.node()


def test_checkpoint_table_rows_empty_history():
    tables = checkpoint_table_rows([])
    assert tables == {"by_val_mse": [], "by_cosine": [], "by_norm_ratio": []}


def test_checkpoint_table_rows_ranks_by_val_mse_ascending():
    history = [
        {"step": 1, "val_mse": 30.0, "val_cos": 0.1, "norm_ratio": 0.5, "variance_ratio": 0.1},
        {"step": 2, "val_mse": 10.0, "val_cos": 0.2, "norm_ratio": 0.9, "variance_ratio": 0.5},
        {"step": 3, "val_mse": 20.0, "val_cos": 0.05, "norm_ratio": 1.5, "variance_ratio": 0.8},
    ]
    tables = checkpoint_table_rows(history)
    assert [r["step"] for r in tables["by_val_mse"]] == [2, 3, 1]


def test_checkpoint_table_rows_ranks_by_cosine_descending():
    history = [
        {"step": 1, "val_mse": 30.0, "val_cos": 0.1, "norm_ratio": 0.5, "variance_ratio": 0.1},
        {"step": 2, "val_mse": 10.0, "val_cos": 0.2, "norm_ratio": 0.9, "variance_ratio": 0.5},
        {"step": 3, "val_mse": 20.0, "val_cos": 0.05, "norm_ratio": 1.5, "variance_ratio": 0.8},
    ]
    tables = checkpoint_table_rows(history)
    assert [r["step"] for r in tables["by_cosine"]] == [2, 1, 3]


def test_checkpoint_table_rows_ranks_by_norm_ratio_closest_to_one():
    history = [
        {"step": 1, "val_mse": 30.0, "val_cos": 0.1, "norm_ratio": 0.5, "variance_ratio": 0.1},
        {"step": 2, "val_mse": 10.0, "val_cos": 0.2, "norm_ratio": 0.9, "variance_ratio": 0.5},
        {"step": 3, "val_mse": 20.0, "val_cos": 0.05, "norm_ratio": 1.5, "variance_ratio": 0.8},
    ]
    tables = checkpoint_table_rows(history)
    assert [r["step"] for r in tables["by_norm_ratio"]] == [2, 1, 3]  # |0.9-1|=0.1 < |0.5-1|=0.5 < |1.5-1|=0.5 (tie broken by order)


def test_checkpoint_table_rows_caps_at_ten():
    history = [
        {"step": i, "val_mse": float(i), "val_cos": 0.0, "norm_ratio": 1.0, "variance_ratio": 1.0}
        for i in range(25)
    ]
    tables = checkpoint_table_rows(history)
    assert len(tables["by_val_mse"]) == 10
    assert len(tables["by_cosine"]) == 10
    assert len(tables["by_norm_ratio"]) == 10
