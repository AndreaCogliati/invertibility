"""Content-addressed feature cache (CLAUDE.md Resource Discipline).

Cache entries are addressed by (audio checksum, MuQ checkpoint id, preprocessing
config, representation name, dtype) -- never by track_id alone -- so a change to
any of those invalidates old entries instead of silently reusing stale ones.
Writes are atomic (temp file + os.replace) so a crash mid-write can never leave
a corrupt cache entry that looks valid.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


@dataclasses.dataclass(frozen=True)
class CacheKeyInputs:
    audio_checksum: str
    muq_model_id: str
    preprocessing: dict[str, Any]
    representation_name: str
    dtype: str


def compute_cache_key(inputs: CacheKeyInputs) -> str:
    """Deterministic hex digest identifying one cache entry."""
    payload = "|".join(
        [
            inputs.audio_checksum,
            inputs.muq_model_id,
            repr(sorted(inputs.preprocessing.items())),
            inputs.representation_name,
            inputs.dtype,
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_path(cache_dir: Path, key: str) -> Path:
    """Shard entries into subdirectories by the key's first two hex chars."""
    return cache_dir / key[:2] / f"{key}.pt"


def save_atomic(path: Path, features: torch.Tensor, metadata: dict[str, Any]) -> None:
    """Write {features, metadata} to `path` atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save({"features": features.cpu(), "metadata": metadata}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load(path: Path) -> tuple[torch.Tensor, dict[str, Any]] | None:
    """Load a cache entry, or None if it doesn't exist. Raises if it exists but is corrupt."""
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["features"], payload["metadata"]
