"""Backend-neutral device handling: the only module in this project that talks to
`torch.cuda`, `torch.xpu`, or `torch.mps` directly.

Supported backends: NVIDIA CUDA, Intel XPU (e.g. Data Center GPU Max / PVC), Apple MPS, CPU.
`auto` prefers CUDA, then XPU, then MPS. CPU is chosen by `auto` only when the caller explicitly
allows it (CLAUDE.md rule 10: never fall back to CPU silently); `--device cpu` is always allowed.

Availability is queried from the installed PyTorch at call time (never cached, never via
CUDA-only packages), so the module imports cleanly on any build and tests can monkeypatch it.
Everything runs in the default FP32; nothing here enables autocast or mixed precision.
"""

from __future__ import annotations

import contextlib
import os
import platform
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

ACCELERATOR_PRIORITY = ("cuda", "xpu", "mps")
DEVICE_CHOICES = ("auto", "cuda", "xpu", "mps", "cpu")
MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"


class DeviceSelectionError(RuntimeError):
    """Raised when the requested device is unavailable or would require a silent CPU fallback."""


def _xpu_module():
    return getattr(torch, "xpu", None)


def backend_available(backend: str) -> bool:
    if backend == "cuda":
        return torch.cuda.is_available()
    if backend == "xpu":
        xpu = _xpu_module()
        return bool(xpu is not None and xpu.is_available())
    if backend == "mps":
        return torch.backends.mps.is_available()
    if backend == "cpu":
        return True
    raise ValueError(f"Unknown backend: {backend!r}")


def availability() -> dict[str, bool]:
    return {b: backend_available(b) for b in ACCELERATOR_PRIORITY}


def _check_no_silent_mps_fallback() -> None:
    # PYTORCH_ENABLE_MPS_FALLBACK=1 makes unsupported MPS ops run on the CPU without telling us,
    # silently splitting the graph across devices.
    if os.environ.get(MPS_FALLBACK_ENV, "0") not in ("", "0"):
        raise DeviceSelectionError(
            f"{MPS_FALLBACK_ENV}={os.environ[MPS_FALLBACK_ENV]!r} would move unsupported MPS ops to the CPU "
            f"silently. Unset it; unsupported ops should fail loudly."
        )


def resolve_device(requested: str, allow_cpu_fallback: bool = False) -> torch.device:
    """Resolve `requested` (auto|cuda|xpu|mps|cpu) to a torch.device.

    - Explicit accelerator requests fail if that backend is unavailable (`allow_cpu_fallback`
      doesn't apply to them).
    - `auto` picks the first available of CUDA, XPU, MPS; with none available it returns CPU only
      if `allow_cpu_fallback` is True, otherwise raises.
    """
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device request: {requested!r} (choose from {', '.join(DEVICE_CHOICES)})")
    avail = availability()
    if requested == "cpu":
        resolved = "cpu"
    elif requested == "auto":
        resolved = next((b for b in ACCELERATOR_PRIORITY if avail[b]), None)
        if resolved is None:
            if not allow_cpu_fallback:
                raise DeviceSelectionError(
                    f"device='auto' found no accelerator ({avail}) and CPU fallback is not permitted. "
                    f"Pass --device cpu explicitly, or set allow_cpu_fallback: true in the config."
                )
            resolved = "cpu"
    else:
        if not avail[requested]:
            raise DeviceSelectionError(
                f"Requested device={requested!r} but it is not available in this PyTorch build/machine "
                f"(torch {torch.__version__}; availability {avail})."
            )
        resolved = requested
    if resolved == "mps":
        _check_no_silent_mps_fallback()
    return torch.device(resolved)


def seed_everything(seed: int, device: torch.device) -> dict[str, Any]:
    """Seed Python, NumPy, torch's CPU generator, and the selected accelerator's generator.
    Calls happen in the same order as the project's earlier `random.seed; np.random.seed;
    torch.manual_seed` sequence, so CPU-side random streams (e.g. parameter initialization) are
    unchanged. Returns a record of what was seeded."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seeded = ["python", "numpy", "torch_cpu"]
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        seeded.append("cuda")
    elif device.type == "xpu":
        _xpu_module().manual_seed_all(seed)
        seeded.append("xpu")
    elif device.type == "mps":
        torch.mps.manual_seed(seed)
        seeded.append("mps")
    return {"seed": seed, "seeded_generators": seeded}


def synchronize(device: torch.device) -> None:
    """Wait for queued work on `device` (no-op on CPU). Use before timing on async backends."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        _xpu_module().synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_stats(device: torch.device) -> dict[str, Any]:
    """Backend-specific allocator statistics in bytes, under backend-neutral keys where possible."""
    if device.type == "cuda":
        return {"allocated_bytes": torch.cuda.memory_allocated(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if device.type == "xpu":
        xpu = _xpu_module()
        return {"allocated_bytes": xpu.memory_allocated(device), "peak_allocated_bytes": xpu.max_memory_allocated(device)}
    if device.type == "mps":
        return {"allocated_bytes": torch.mps.current_allocated_memory(),
                "driver_allocated_bytes": torch.mps.driver_allocated_memory()}
    return {}


def device_count(backend: str) -> int:
    """Number of visible devices of an accelerator backend (0 when unavailable)."""
    if not backend_available(backend):
        return 0
    if backend == "cuda":
        return torch.cuda.device_count()
    if backend == "xpu":
        return _xpu_module().device_count()
    return 1


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "xpu":
        return _xpu_module().get_device_name(device)
    if device.type == "mps":
        return f"Apple MPS ({platform.machine()}, {platform.platform()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def device_metadata(requested: str, device: torch.device) -> dict[str, Any]:
    """Fields recorded in logs, run manifests, checkpoints, and W&B for every run."""
    avail = availability()
    return {
        "requested_device": requested,
        "resolved_device": str(device),
        "backend": device.type,
        "device_name": device_name(device),
        "dtype": str(torch.get_default_dtype()).replace("torch.", ""),
        "torch_version": torch.__version__,
        "accelerator_available": avail,
        "cuda_device_count": device_count("cuda"),
        "xpu_device_count": device_count("xpu"),
        "mps_fallback_env": os.environ.get(MPS_FALLBACK_ENV),
    }


def require_fp32_default() -> None:
    """Experiments run in FP32 on every backend; refuse to start if something changed the default."""
    if torch.get_default_dtype() != torch.float32:
        raise DeviceSelectionError(f"default dtype is {torch.get_default_dtype()}, expected torch.float32")


def load_portable(path: Path) -> dict:
    """Load a checkpoint saved on any backend (MPS/CUDA/XPU/CPU tensors) onto the CPU; callers
    then `load_state_dict` into a model already on the target device."""
    return torch.load(path, map_location="cpu", weights_only=False)


@contextlib.contextmanager
def backend_stage(stage: str, device: torch.device) -> Iterator[None]:
    """Annotate any failure inside `stage` with the stage and device, without changing the
    exception type or catching it. Unsupported-operator errors on XPU/MPS therefore surface with
    context instead of being retried elsewhere."""
    try:
        yield
    except Exception as exc:
        exc.add_note(f"[{stage}] failed on device {device} (backend {device.type}, torch {torch.__version__}).")
        if device.type in ("xpu", "mps"):
            exc.add_note(
                "If this is an unsupported-operator error, the op is not implemented for this backend in this "
                "PyTorch build. No CPU fallback is attempted; report the op name above."
            )
        raise
