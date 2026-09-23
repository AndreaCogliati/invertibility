"""Weights & Biases integration for `tasks/TASK-03.md`.

Kept as a thin, isolated wrapper around the `wandb` SDK so the training script's core logic
(unchanged from TASK-02, per this task's "do not modify the scientific formulation" instruction)
stays readable, and so this module's non-network-touching pieces (run-name generation, table
building, top-K selection) are independently unit-testable without a live W&B connection or
network access.
"""

from __future__ import annotations

import datetime
import platform
from pathlib import Path
from typing import Any

PROJECT = "muq-invertibility"
GROUP = "spectrogram-control"


def make_run_name(prefix: str = "spec_dense_gtzan", now: datetime.datetime | None = None) -> str:
    """`spec_dense_gtzan_<timestamp>`, per this task's suggested naming scheme. `now` is
    injectable for deterministic tests; defaults to the real current UTC time."""
    now = now or datetime.datetime.now(datetime.UTC)
    return f"{prefix}_{now.strftime('%Y%m%dT%H%M%SZ')}"


def build_run_metadata(environment: dict[str, Any], job_id: str | None) -> dict[str, Any]:
    """Fields this task requires logged 'at run start': git commit, hostname, machine type,
    BlueHive job ID. `environment` is `logging_utils.capture_environment_manifest(...)`'s output
    -- reused rather than re-derived, so this file carries the exact same reproducibility fields
    every other stage in this project already captures (CLAUDE.md rule 2)."""
    return {
        "git_commit": environment.get("git_commit"),
        "git_dirty": environment.get("git_dirty"),
        "hostname": environment.get("hostname", platform.node()),
        "machine_type": environment.get("machine"),
        "platform": environment.get("platform"),
        "cuda_available": environment.get("cuda_available"),
        "cuda_device_name": environment.get("cuda_device_name"),
        "mps_available": environment.get("mps_available"),
        "slurm_job_id": job_id or environment.get("slurm_job_id"),
    }


def init_run(
    *,
    config: dict[str, Any],
    environment: dict[str, Any],
    job_id: str | None = None,
    run_name: str | None = None,
    mode: str = "online",
    group: str = GROUP,
    job_type: str | None = None,
    tags: list[str] | None = None,
    resume_id: str | None = None,
):
    """Initializes a W&B run under the fixed `PROJECT`, with `group` defaulting to TASK-03's
    `spectrogram-control` (overridable -- e.g. `tasks/TASK-04.md`'s `spectrogram-gate3`). Import
    of `wandb` is local to this function so the rest of the codebase (and its test suite) never
    requires `wandb` to be importable or configured -- only the scripts that actually opt into
    W&B logging pay that cost. `mode='offline'` or `'disabled'` lets a caller (or a BlueHive
    compute node with no outbound network access) run without a live connection; `wandb sync` can
    push an offline run later.

    `resume_id` (`tasks/TASK-14.md`'s "resumption path must append to the same scientific run
    lineage" requirement): when a caller resumes training from a checkpoint that recorded a prior
    `run.id`, passing that id here reopens the SAME W&B run (`resume="must"`) instead of creating
    a new one with a fresh timestamped name -- `run_name`/`group`/`job_type`/`tags` are then
    ignored by W&B for an existing run (its original values are kept), consistent with "resuming
    must append to the existing run history rather than creating ambiguous duplicate runs"
    (CLAUDE.md's Checkpointing and Resumption section, generalized here to W&B runs)."""
    import wandb

    if resume_id is not None:
        return wandb.init(project=PROJECT, id=resume_id, resume="must", mode=mode)

    run_name = run_name or make_run_name()
    full_config = {**config, **build_run_metadata(environment, job_id)}
    return wandb.init(
        project=PROJECT, group=group, job_type=job_type, tags=tags, name=run_name, config=full_config, mode=mode
    )


def checkpoint_table_rows(
    history: list[dict[str, float]],
) -> dict[str, list[dict[str, float]]]:
    """Builds the three top-10 tables this task requires (by val MSE, by cosine, by norm ratio),
    per checkpoint record. `history` entries must carry `step`, `val_mse`, `val_cos`,
    `norm_ratio`, `variance_ratio`. Pure function -- the caller turns the result into
    `wandb.Table`s; kept out of this function so it's testable without `wandb` installed.
    """
    if not history:
        return {"by_val_mse": [], "by_cosine": [], "by_norm_ratio": []}
    return {
        "by_val_mse": sorted(history, key=lambda r: r["val_mse"])[:10],
        "by_cosine": sorted(history, key=lambda r: -r["val_cos"])[:10],
        "by_norm_ratio": sorted(history, key=lambda r: abs(r["norm_ratio"] - 1.0))[:10],
    }


def log_checkpoint_tables(run, history: list[dict[str, float]]) -> None:
    """Logs the three top-10 W&B tables this task requires to an active run."""
    import wandb

    tables = checkpoint_table_rows(history)
    columns = ["step", "val_mse", "val_cos", "norm_ratio", "variance_ratio"]
    for name, rows in tables.items():
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row.get(c) for c in columns])
        run.log({f"checkpoint_tables/{name}": table})


def log_audio_examples(run, step: int, examples: dict[str, tuple[Any, int]]) -> None:
    """`examples`: {label -> (waveform_1d_numpy_or_path, sample_rate)}. Logs each as
    `wandb.Audio` under a fixed key per label, so W&B's UI lets you scrub through training steps
    for the same held-out example ('listening to training evolution', per this task)."""
    import wandb

    run.log(
        {f"audio/{label}": wandb.Audio(data, sample_rate=sr) for label, (data, sr) in examples.items()},
        step=step,
    )


def log_artifact_file(run, path: Path, name: str, artifact_type: str) -> None:
    """Uploads one local file (checkpoint, report, manifest, config) as a versioned W&B
    artifact, per this task's 'the experiment should be fully reproducible from W&B'
    requirement."""
    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_file(str(path))
    run.log_artifact(artifact)
