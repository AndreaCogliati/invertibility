"""Backend handling in `muq_inversion.device` and the TASK-15 code paths that use it.

CUDA, XPU, and MPS availability is mocked, so every selection path is tested on any machine.
Checkpoints "saved on" CUDA/XPU are simulated by tagging storages with that location at save time.
"""

from __future__ import annotations

import re
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from muq_inversion import device as dev
from muq_inversion.device import DeviceSelectionError, resolve_device
from muq_inversion.models.muq_mel_translator import MuQToMelTranslator
from muq_inversion.time_alignment import source_positions

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def backends(monkeypatch: pytest.MonkeyPatch):
    """Returns a setter: backends(cuda=..., xpu=..., mps=...) mocks availability and the
    per-backend calls `device.py` makes, recording seeding calls in `calls`."""
    calls: list[tuple[str, int]] = []
    monkeypatch.delenv(dev.MPS_FALLBACK_ENV, raising=False)

    def set_(*, cuda: bool = False, xpu: bool = False, mps: bool = False):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2 if cuda else 0)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "Fake NVIDIA A100")
        monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda s: calls.append(("cuda", s)))
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        monkeypatch.setattr(torch.mps, "manual_seed", lambda s: calls.append(("mps", s)))
        xpu_fns = {"is_available": lambda: xpu, "device_count": lambda: 4 if xpu else 0,
                   "get_device_name": lambda *_: "Fake Intel Data Center GPU Max 1550",
                   "manual_seed_all": lambda s: calls.append(("xpu", s))}
        if hasattr(torch, "xpu"):  # patch the real module so torch.manual_seed's own XPU hook keeps working
            for name, fn in xpu_fns.items():
                monkeypatch.setattr(torch.xpu, name, fn)
        else:
            monkeypatch.setattr(torch, "xpu", types.SimpleNamespace(**xpu_fns), raising=False)
        return calls

    return set_


# --- automatic selection priority ---------------------------------------------------------

@pytest.mark.parametrize(
    ("cuda", "xpu", "mps", "expected"),
    [
        (True, True, True, "cuda"),
        (True, False, True, "cuda"),
        (False, True, True, "xpu"),
        (False, True, False, "xpu"),
        (False, False, True, "mps"),
    ],
)
def test_auto_priority_is_cuda_then_xpu_then_mps(backends, cuda, xpu, mps, expected):
    backends(cuda=cuda, xpu=xpu, mps=mps)
    assert resolve_device("auto").type == expected


def test_auto_without_accelerator_refuses_cpu_unless_permitted(backends):
    backends()
    with pytest.raises(DeviceSelectionError, match="no accelerator"):
        resolve_device("auto")
    assert resolve_device("auto", allow_cpu_fallback=True).type == "cpu"


def test_xpu_missing_from_torch_build_is_just_unavailable(backends, monkeypatch):
    backends(mps=True)
    monkeypatch.delattr(torch, "xpu", raising=False)
    assert dev.availability()["xpu"] is False
    assert resolve_device("auto").type == "mps"


# --- explicit selection and unavailable-backend errors ------------------------------------

@pytest.mark.parametrize("backend", ["cuda", "xpu", "mps"])
def test_explicit_backend_is_selected_when_available(backends, backend):
    backends(cuda=True, xpu=True, mps=True)
    assert resolve_device(backend).type == backend


@pytest.mark.parametrize("backend", ["cuda", "xpu", "mps"])
def test_explicit_backend_unavailable_raises_even_with_cpu_fallback(backends, backend):
    others = {b: True for b in ("cuda", "xpu", "mps") if b != backend}
    backends(**others)
    with pytest.raises(DeviceSelectionError, match=f"device='{backend}'"):
        resolve_device(backend, allow_cpu_fallback=True)


def test_explicit_cpu_always_allowed_and_unknown_rejected(backends):
    backends(cuda=True)
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


def test_mps_with_silent_cpu_fallback_env_is_refused(backends, monkeypatch):
    backends(mps=True)
    monkeypatch.setenv(dev.MPS_FALLBACK_ENV, "1")
    with pytest.raises(DeviceSelectionError, match=dev.MPS_FALLBACK_ENV):
        resolve_device("mps")
    monkeypatch.setenv(dev.MPS_FALLBACK_ENV, "0")
    assert resolve_device("mps").type == "mps"


def test_logging_utils_select_device_delegates(backends):
    from muq_inversion.logging_utils import select_device

    backends(xpu=True, mps=True)
    assert select_device("auto", allow_cpu_fallback=False).type == "xpu"


# --- seeding and metadata ------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["cuda", "xpu", "mps"])
def test_seed_everything_seeds_cpu_and_selected_backend(backends, backend):
    calls = backends(cuda=True, xpu=True, mps=True)
    record = dev.seed_everything(7, torch.device(backend))
    assert record == {"seed": 7, "seeded_generators": ["python", "numpy", "torch_cpu", backend]}
    assert (backend, 7) in calls


def test_seed_everything_matches_previous_cpu_seeding_sequence():
    import random

    dev.seed_everything(0, torch.device("cpu"))
    a = (random.random(), np.random.rand(), torch.rand(3))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    b = (random.random(), np.random.rand(), torch.rand(3))
    assert a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])


@pytest.mark.parametrize(("backend", "name"), [("cuda", "Fake NVIDIA A100"), ("xpu", "Fake Intel Data Center GPU Max 1550")])
def test_device_metadata_records_request_resolution_and_availability(backends, backend, name):
    backends(cuda=True, xpu=True)
    meta = dev.device_metadata("auto", torch.device(backend))
    assert meta["requested_device"] == "auto"
    assert meta["resolved_device"] == backend and meta["backend"] == backend
    assert meta["device_name"] == name
    assert meta["dtype"] == "float32"
    assert meta["torch_version"] == torch.__version__
    assert meta["accelerator_available"] == {"cuda": True, "xpu": True, "mps": False}
    assert meta["cuda_device_count"] == 2 and meta["xpu_device_count"] == 4


def test_environment_manifest_and_wandb_metadata_include_xpu(backends):
    from muq_inversion.logging_utils import capture_environment_manifest
    from muq_inversion.wandb_logging import build_run_metadata

    backends(xpu=True)
    env = capture_environment_manifest(REPO_ROOT)
    assert env["xpu_available"] is True and env["xpu_device_count"] == 4
    assert env["xpu_device_name"] == "Fake Intel Data Center GPU Max 1550"
    meta = build_run_metadata(env, None)
    assert meta["xpu_available"] is True and meta["xpu_device_name"].startswith("Fake Intel")


def test_require_fp32_default():
    dev.require_fp32_default()
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(DeviceSelectionError, match="float64"):
            dev.require_fp32_default()
    finally:
        torch.set_default_dtype(old)


def test_backend_stage_annotates_without_swallowing():
    with pytest.raises(NotImplementedError) as info, dev.backend_stage("BigVGAN inference", torch.device("xpu")):
        raise NotImplementedError("aten::foo not implemented for 'XPU'")
    notes = " ".join(info.value.__notes__)
    assert "[BigVGAN inference]" in notes and "xpu" in notes and "No CPU fallback" in notes


# --- checkpoint loading across devices -----------------------------------------------------

@pytest.mark.parametrize("saved_on", ["cuda:0", "xpu:0", "mps:0"])
def test_checkpoint_saved_on_any_accelerator_loads_on_cpu(tmp_path, monkeypatch, saved_on):
    model = MuQToMelTranslator(16, 8, 5)
    original = torch.serialization.location_tag
    monkeypatch.setattr(torch.serialization, "location_tag", lambda storage: saved_on)
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    monkeypatch.setattr(torch.serialization, "location_tag", original)
    assert saved_on.encode() in path.read_bytes()  # the storages really are tagged as `saved_on`

    state = dev.load_portable(path)
    assert all(v.device.type == "cpu" for v in state["model_state_dict"].values())
    restored = MuQToMelTranslator(16, 8, 5)
    restored.load_state_dict(state["model_state_dict"])
    for a, b in zip(model.state_dict().values(), restored.state_dict().values(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a real MPS device")
def test_checkpoint_round_trip_between_mps_and_cpu(tmp_path):
    model = MuQToMelTranslator(16, 8, 5).to("mps")
    path = tmp_path / "mps.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    on_cpu = MuQToMelTranslator(16, 8, 5)
    on_cpu.load_state_dict(dev.load_portable(path)["model_state_dict"])
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        assert torch.allclose(on_cpu(x, 20), model(x.to("mps"), 20).cpu(), atol=1e-5)
    torch.save({"model_state_dict": on_cpu.state_dict()}, path)
    back = MuQToMelTranslator(16, 8, 5).to("mps")
    back.load_state_dict(dev.load_portable(path)["model_state_dict"])
    assert all(p.device.type == "mps" for p in back.parameters())


# --- CPU forward/backward/checkpoint cycle -------------------------------------------------

def test_cpu_forward_backward_checkpoint_resume_cycle(tmp_path):
    """Corrected-alignment translator on CPU: 3 steps, checkpoint, 2 more steps after a portable
    reload must equal 5 uninterrupted steps exactly."""
    device = resolve_device("cpu")
    feats, target = torch.randn(2, 50, 32), torch.randn(2, 10, 187)
    positions = source_positions(np.arange(187) * 256.0 + 128.0, np.arange(50) * 960.0).to(device)

    def make():
        dev.seed_everything(0, device)
        m = MuQToMelTranslator(32, 16, 10).to(device)
        return m, torch.optim.Adam(m.parameters(), lr=1e-3)

    def step(m, opt):
        loss = (m(feats, 187, positions) - target).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    ref, ref_opt = make()
    ref_losses = [step(ref, ref_opt) for _ in range(5)]

    m, opt = make()
    losses = [step(m, opt) for _ in range(3)]
    path = tmp_path / "cycle.pt"
    torch.save({"model_state_dict": m.state_dict(), "optimizer_state_dict": opt.state_dict(),
                "device_metadata": dev.device_metadata("cpu", device)}, path)
    state = dev.load_portable(path)
    m2, opt2 = make()
    m2.load_state_dict(state["model_state_dict"])
    opt2.load_state_dict(state["optimizer_state_dict"])
    losses += [step(m2, opt2) for _ in range(2)]

    assert losses == ref_losses
    assert losses[-1] < losses[0]
    assert state["device_metadata"]["backend"] == "cpu"
    for a, b in zip(ref.parameters(), m2.parameters(), strict=True):
        assert torch.equal(a, b)


# --- no unconditional backend calls in TASK-15 paths ---------------------------------------

TASK15_PATHS = [
    "scripts/train_task15_bigvgan_memorization.py",
    "scripts/task15_bigvgan_oracle.py",
    "scripts/task15_smoke_test.py",
    "scripts/task16_compare_n2.py",
    "scripts/fetch_bigvgan.py",
    "src/muq_inversion/bigvgan_common.py",
    "src/muq_inversion/time_alignment.py",
    "src/muq_inversion/models/muq_mel_translator.py",
    "src/muq_inversion/encoder.py",
    "src/muq_inversion/logging_utils.py",
    "src/muq_inversion/wandb_logging.py",
]
FORBIDDEN = re.compile(
    r"torch\.cuda\.|torch\.xpu\.|torch\.mps\.|\.cuda\(|\.xpu\(|autocast|GradScaler|\.half\(|bfloat16"
    r"|map_location\s*=\s*['\"](cuda|xpu|mps)|\.to\(\s*['\"](cuda|xpu|mps)|device\s*=\s*['\"](cuda|xpu|mps)"
)


@pytest.mark.parametrize("rel", TASK15_PATHS)
def test_task15_paths_have_no_direct_backend_calls(rel):
    hits = [f"{rel}:{i}: {line.strip()}" for i, line in enumerate((REPO_ROOT / rel).read_text().splitlines(), 1)
            if FORBIDDEN.search(line.split("#")[0])]
    assert not hits, "backend-specific calls belong in muq_inversion/device.py:\n" + "\n".join(hits)


def test_bigvgan_is_loaded_without_cuda_kernel():
    src = (REPO_ROOT / "src/muq_inversion/bigvgan_common.py").read_text()
    assert "use_cuda_kernel=False" in src and "use_cuda_kernel=True" not in src
