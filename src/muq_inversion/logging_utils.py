"""Device selection and run-manifest metadata capture.

Shared by every CLI stage so that reproducibility fields (CLAUDE.md rule 2)
are recorded identically everywhere.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy
import torch

from muq_inversion.device import (
    DeviceSelectionError,
    availability,
    device_count,
    device_name,
    resolve_device,
)

__all__ = ["DeviceSelectionError", "capture_environment_manifest", "select_device"]


def select_device(requested: str, allow_cpu_fallback: bool) -> torch.device:
    """Resolve `requested` ("auto", "cuda", "xpu", "mps", or "cpu") to a torch.device.

    Thin wrapper kept for the many existing callers; the logic lives in
    `muq_inversion.device.resolve_device` ("auto" = CUDA, then XPU, then MPS; never a silent
    CPU fallback, CLAUDE.md rule 10).
    """
    return resolve_device(requested, allow_cpu_fallback)


def _git_commit_info(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None

    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty) if dirty is not None else None,
    }


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


PACKAGE_VERSIONS_TRACKED = (
    "torch",
    "torchaudio",
    "numpy",
    "muq",
    "transformers",
    "librosa",
    "soundfile",
    "huggingface_hub",
)


def capture_environment_manifest(repo_root: Path) -> dict[str, Any]:
    """Collect the fields CLAUDE.md requires in every run manifest."""
    avail = availability()
    mps_available = avail["mps"]
    mps_built = torch.backends.mps.is_built()
    cuda_available = avail["cuda"]
    xpu_available = avail["xpu"]

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "numpy_version": numpy.__version__,
        "mps_available": mps_available,
        "mps_built": mps_built,
        "cuda_available": cuda_available,
        "cuda_device_name": device_name(torch.device("cuda", 0)) if cuda_available else None,
        "cuda_device_count": device_count("cuda"),
        "xpu_available": xpu_available,
        "xpu_device_name": device_name(torch.device("xpu", 0)) if xpu_available else None,
        "xpu_device_count": device_count("xpu"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "package_versions": {
            name: _package_version(name) for name in PACKAGE_VERSIONS_TRACKED
        },
        **_git_commit_info(repo_root),
    }


@dataclasses.dataclass(frozen=True)
class RunManifest:
    stage: str
    config_name: str
    resolved_config: dict[str, Any]
    environment: dict[str, Any]
    device: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
