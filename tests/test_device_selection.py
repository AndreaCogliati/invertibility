"""Tests the CUDA/MPS-agnostic device selection added for `tasks/TASK-03.md` (BlueHive uses
CUDA; this machine uses MPS). Uses monkeypatch to simulate both backends without needing real
CUDA hardware -- this machine only has MPS, so these are the only tests that exercise the CUDA
branches at all.
"""

from __future__ import annotations

import pytest
import torch

from muq_inversion.logging_utils import DeviceSelectionError, select_device


def _patch_backends(monkeypatch: pytest.MonkeyPatch, *, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    if hasattr(torch, "xpu"):  # keep these CUDA/MPS cases hermetic on XPU hosts too
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)


def test_auto_prefers_cuda_over_mps(monkeypatch):
    _patch_backends(monkeypatch, cuda=True, mps=True)
    assert select_device("auto", allow_cpu_fallback=False).type == "cuda"


def test_auto_falls_back_to_mps_when_no_cuda(monkeypatch):
    _patch_backends(monkeypatch, cuda=False, mps=True)
    assert select_device("auto", allow_cpu_fallback=False).type == "mps"


def test_auto_raises_when_neither_available_and_no_cpu_fallback(monkeypatch):
    _patch_backends(monkeypatch, cuda=False, mps=False)
    with pytest.raises(DeviceSelectionError):
        select_device("auto", allow_cpu_fallback=False)


def test_auto_falls_back_to_cpu_when_explicitly_permitted(monkeypatch):
    _patch_backends(monkeypatch, cuda=False, mps=False)
    assert select_device("auto", allow_cpu_fallback=True).type == "cpu"


def test_explicit_cuda_request_raises_when_unavailable(monkeypatch):
    _patch_backends(monkeypatch, cuda=False, mps=True)
    with pytest.raises(DeviceSelectionError):
        select_device("cuda", allow_cpu_fallback=True)  # allow_cpu_fallback must not matter here


def test_explicit_mps_request_raises_when_unavailable(monkeypatch):
    _patch_backends(monkeypatch, cuda=True, mps=False)
    with pytest.raises(DeviceSelectionError):
        select_device("mps", allow_cpu_fallback=True)


def test_explicit_cuda_request_succeeds_when_available(monkeypatch):
    _patch_backends(monkeypatch, cuda=True, mps=False)
    assert select_device("cuda", allow_cpu_fallback=False).type == "cuda"


def test_cpu_request_always_succeeds(monkeypatch):
    _patch_backends(monkeypatch, cuda=False, mps=False)
    assert select_device("cpu", allow_cpu_fallback=False).type == "cpu"


def test_unknown_device_string_raises_value_error():
    with pytest.raises(ValueError):
        select_device("tpu", allow_cpu_fallback=True)
