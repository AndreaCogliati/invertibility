"""Device selection and run-manifest metadata capture.

Shared by every CLI stage so that reproducibility fields (CLAUDE.md rule 2)
are recorded identically everywhere.
"""

from __future__ import annotations

import dataclasses
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy
import torch


class DeviceSelectionError(RuntimeError):
    """Raised when the requested device is unavailable and fallback is not permitted."""


def select_device(requested: str, allow_cpu_fallback: bool) -> torch.device:
    """Resolve `requested` ("auto", "mps", or "cpu") to a torch.device.

    Never silently falls back from MPS to CPU (CLAUDE.md rule 10): if MPS is
    requested or chosen by "auto" but unavailable, this raises unless
    `allow_cpu_fallback` is explicitly True.
    """
    mps_available = torch.backends.mps.is_available()

    if requested == "cpu":
        return torch.device("cpu")

    if requested in ("auto", "mps"):
        if mps_available:
            return torch.device("mps")
        if requested == "mps" or not allow_cpu_fallback:
            raise DeviceSelectionError(
                f"Requested device={requested!r} but MPS is not available on this "
                "machine and allow_cpu_fallback is False. Set environment."
                "allow_cpu_fallback: true in the config to permit CPU fallback."
            )
        return torch.device("cpu")

    raise ValueError(f"Unknown device request: {requested!r}")


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
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "numpy_version": numpy.__version__,
        "mps_available": mps_available,
        "mps_built": mps_built,
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
