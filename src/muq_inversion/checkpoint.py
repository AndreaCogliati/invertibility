"""Atomic, resumable checkpointing (CLAUDE.md Checkpointing and Resumption).

Gate 3 only needs to *write* a valid checkpoint at the end of a tiny-set
overfit run; validating checkpoint/resume numerical equivalence is a Gate 4
deliverable (CLAUDE.md's own gate ordering). This module still records the
full state list Gate 4 will need, so no checkpoint written here has to be
redone later.
"""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig, config_to_dict
from .logging_utils import capture_environment_manifest


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checkpoint_state(
    *,
    config: ExperimentConfig,
    repo_root: Path,
    manifest_path: Path,
    generator: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    samples_seen: int,
    best_metric: float | None,
) -> dict[str, Any]:
    """Assemble the full checkpoint payload required by CLAUDE.md's Checkpointing section."""
    torch_rng_state = torch.get_rng_state()
    mps_rng_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None

    return {
        "generator_state_dict": generator.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "samples_seen": samples_seen,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch_rng_state,
        "torch_mps_rng_state": mps_rng_state,
        "resolved_config": config_to_dict(config),
        "manifest_hash": _file_hash(manifest_path),
        "uv_lock_hash": _file_hash(repo_root / "uv.lock"),
        "environment": capture_environment_manifest(repo_root),
        "best_metric": best_metric,
    }


def save_checkpoint_atomic(path: Path, state: dict[str, Any]) -> None:
    """Write `state` to `path` atomically (temp file + os.replace), then re-load to validate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    # Validate immediately after writing, per CLAUDE.md: reload and check key tensors/metadata.
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = {
        "generator_state_dict",
        "optimizer_state_dict",
        "step",
        "resolved_config",
        "manifest_hash",
        "uv_lock_hash",
    }
    missing = required_keys - reloaded.keys()
    if missing:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checkpoint at {path} failed post-write validation (missing keys: {missing}); removed."
        )


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if state.get("torch_mps_rng_state") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["torch_mps_rng_state"])
