"""Tests for TASK-15's BigVGAN-v2 mel pipeline helpers and translator.

The BigVGAN-dependent tests are skipped if `scripts/fetch_bigvgan.py` hasn't been run (the
official code and checkpoint live in git-ignored `third_party/`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from muq_inversion.bigvgan_common import (
    BIGVGAN_DIR,
    load_mel_crops,
    mel_l1,
    official_mel,
    shuffled_across_pairing,
)
from muq_inversion.models.muq_mel_translator import MuQToMelTranslator

REPO_ROOT = Path(__file__).resolve().parents[1]
needs_bigvgan = pytest.mark.skipif(not (BIGVGAN_DIR / "config.json").exists(), reason="run scripts/fetch_bigvgan.py")


@pytest.fixture(scope="module")
def h():
    import sys

    sys.path.insert(0, str(BIGVGAN_DIR))
    from env import AttrDict

    return AttrDict(json.loads((BIGVGAN_DIR / "config.json").read_text()))


@needs_bigvgan
def test_checkpoint_mel_hyperparameters_are_what_task15_records(h):
    """Guards against a silently different checkpoint/config: these values are documented in
    configs/task15_bigvgan_memorization.yaml and the TASK-15 report."""
    assert (h.sampling_rate, h.num_mels, h.n_fft, h.hop_size, h.win_size, h.fmin, h.fmax) == (24000, 100, 1024, 256, 1024, 0, None)


@needs_bigvgan
def test_official_mel_has_one_frame_per_hop(h):
    for hops in (2, 50, 187):  # >= 2: official reflect pad is 384 samples
        mel = official_mel(torch.randn(1, hops * h.hop_size) * 0.1, h)
        assert mel.shape == (1, h.num_mels, hops)


@needs_bigvgan
def test_official_mel_matches_known_log_floor(h):
    mel = official_mel(torch.zeros(1, 10 * h.hop_size), h)
    assert torch.allclose(mel, torch.full_like(mel, float(torch.log(torch.tensor(1e-5)))), atol=1e-3)


@needs_bigvgan
def test_load_mel_crops_n2_geometry(h):
    crops = load_mel_crops(REPO_ROOT / "data/manifests/gate3_scale_2.csv", h, num_hops=187)
    assert [c.track_id for c in crops] == ["pop.00012", "disco.00049"]
    for c in crops:
        assert c.waveform.shape == (1, 187 * 256)
        assert c.target_mel.shape == (1, 100, 187)
        assert float(c.waveform.abs().max()) <= 0.95 + 1e-6  # peak-normalized track, then cropped


def test_translator_output_geometry_and_param_count():
    model = MuQToMelTranslator(input_channels=1024, hidden_channels=256, num_mels=100)
    out = model(torch.randn(2, 50, 1024), num_mel_frames=187)
    assert out.shape == (2, 100, 187)
    assert sum(p.numel() for p in model.parameters()) == 1024 * 256 + 256 + 3 * (256 * 256 * 5 + 256) + 256 * 100 + 100
    assert model.receptive_field_mel_frames == 13


def test_translator_output_depends_on_input():
    torch.manual_seed(0)
    model = MuQToMelTranslator(1024, 32, 100)
    a, b = torch.randn(1, 50, 1024), torch.randn(1, 50, 1024)
    assert not torch.allclose(model(a, 187), model(b, 187))


def test_shuffled_across_pairing_is_a_deterministic_derangement():
    ids = [f"t{i}" for i in range(8)]
    pairing = shuffled_across_pairing(ids)
    assert all(pairing[i] != i for i in ids)
    assert sorted(pairing.values()) == sorted(ids)
    assert pairing == shuffled_across_pairing(ids)
    assert shuffled_across_pairing(["a", "b"]) == {"a": "b", "b": "a"}


def test_shuffled_across_pairing_requires_two_crops():
    with pytest.raises(ValueError):
        shuffled_across_pairing(["only"])


def test_mel_l1():
    a = torch.zeros(1, 100, 10)
    assert mel_l1(a, a + 0.5) == pytest.approx(0.5)
