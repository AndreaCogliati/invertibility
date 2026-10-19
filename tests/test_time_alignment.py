"""Tests for TASK-16's MuQ -> BigVGAN mel time alignment and decode-time mel-floor clamp.

The derived frame centers are checked against the real implementations with impulse responses,
so the formulas in `time_alignment.py` can't silently drift from the code they describe.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from muq_inversion.bigvgan_common import (
    BIGVGAN_DIR,
    bigvgan_log_mel_floor,
    clamp_to_mel_floor,
    official_mel,
)
from muq_inversion.models.muq_mel_translator import MuQToMelTranslator
from muq_inversion.time_alignment import (
    bigvgan_mel_frame_centers,
    interpolate_at_positions,
    muq_token_centers,
    source_positions,
)

needs_bigvgan = pytest.mark.skipif(not (BIGVGAN_DIR / "config.json").exists(), reason="run scripts/fetch_bigvgan.py")
CROP = 187 * 256  # TASK-15/16 crop length


@pytest.fixture(scope="module")
def h():
    import sys

    sys.path.insert(0, str(BIGVGAN_DIR))
    from env import AttrDict

    return AttrDict(json.loads((BIGVGAN_DIR / "config.json").read_text()))


@pytest.fixture(scope="module")
def muq_inner():
    from muq_inversion.encoder import FrozenMuQEncoder

    return FrozenMuQEncoder("OpenMuQ/MuQ-large-msd-iter", torch.device("cpu")).model.model


# ---- derived centers vs the real implementations ----


@needs_bigvgan
def test_bigvgan_frame_centers_formula(h):
    centers = bigvgan_mel_frame_centers(h, 187)
    assert centers.shape == (187,)
    assert centers[0] == 128 and centers[1] - centers[0] == 256 and centers[-1] == 186 * 256 + 128


@needs_bigvgan
def test_bigvgan_frame_centers_match_official_mel_impulse_response(h):
    centers = bigvgan_mel_frame_centers(h, 187)
    for frame in (10, 93, 150):
        imp = torch.zeros(1, CROP)
        imp[0, int(centers[frame])] = 1.0
        energy = official_mel(imp, h)[0].exp().sum(0)
        assert int(energy.argmax()) == frame


def test_muq_token_centers_derived_from_model(muq_inner):
    centers = muq_token_centers(muq_inner, 50)
    assert centers.shape == (50,)
    np.testing.assert_allclose(centers, 960.0 * np.arange(50))


def test_muq_token_centers_match_conv_front_end_impulse_response(muq_inner):
    """A click at a token's derived center moves that token's conv-front-end output most (the
    front end is local; the conformer after it is not, so it's excluded)."""
    centers = muq_token_centers(muq_inner, 50)
    base = torch.randn(1, CROP, generator=torch.Generator().manual_seed(0)) * 1e-3

    def front_end(w):
        with torch.inference_mode():
            f = muq_inner.preprocessing(w, features=["melspec_2048"])
            return muq_inner.conv(muq_inner.normalize(f)["melspec_2048"])

    ref = front_end(base)
    for token in (5, 25, 40):
        w = base.clone()
        w[0, int(centers[token])] += 0.5
        assert int((front_end(w) - ref).norm(dim=-1)[0].argmax()) == token


# ---- mapping, endpoints, shapes ----


@needs_bigvgan
def test_source_positions_endpoints_and_monotonicity(h):
    pos = source_positions(bigvgan_mel_frame_centers(h, 187), 960.0 * np.arange(50))
    assert pos.shape == (187,)
    assert pos[0].item() == pytest.approx(128 / 960)
    assert pos[-1].item() == 49.0  # last mel frame (47,744) lies past the last token (47,040): clamped
    assert torch.all(pos[1:] >= pos[:-1])
    assert int((pos == 49.0).sum()) == 3  # frames 184-186 are centred after the last token
    assert pos[93].item() == pytest.approx((93 * 256 + 128) / 960)


def test_source_positions_clamps_before_first_center():
    pos = source_positions(np.array([-50.0, 0.0, 5.0]), np.array([0.0, 10.0]))
    assert pos.tolist() == pytest.approx([0.0, 0.0, 0.5])


def test_source_positions_rejects_non_increasing_centers():
    with pytest.raises(ValueError):
        source_positions(np.array([1.0]), np.array([0.0, 0.0]))


def test_interpolate_at_positions_exact_and_midpoints():
    x = torch.arange(10.0).reshape(1, 2, 5)  # channel 0: 0..4, channel 1: 5..9
    out = interpolate_at_positions(x, torch.tensor([0.0, 1.5, 4.0]))
    assert out.shape == (1, 2, 3)
    assert out[0, 0].tolist() == pytest.approx([0.0, 1.5, 4.0])
    assert out[0, 1].tolist() == pytest.approx([5.0, 6.5, 9.0])


def test_interpolate_at_positions_rejects_out_of_range():
    with pytest.raises(ValueError):
        interpolate_at_positions(torch.zeros(1, 1, 5), torch.tensor([4.5]))


def test_interpolate_at_positions_reproduces_f_interpolate_given_its_positions():
    """Sanity: with F.interpolate's own (align_corners=False, edge-clamped) source coordinates,
    the explicit interpolation is the same operation -- only the coordinates differ in TASK-16."""
    x = torch.randn(2, 8, 50)
    j = torch.arange(187, dtype=torch.float32)
    pos = ((j + 0.5) * 50 / 187 - 0.5).clamp(0, 49)
    torch.testing.assert_close(interpolate_at_positions(x, pos), F.interpolate(x, size=187, mode="linear", align_corners=False), atol=1e-4, rtol=0)  # float rounding of 50/187


# ---- translator ----


def test_translator_frame_center_path_shape_and_unchanged_param_count():
    torch.manual_seed(0)
    model = MuQToMelTranslator(1024, 256, 100)
    pos = torch.linspace(0, 49, 187)
    out = model(torch.randn(2, 50, 1024), 187, pos)
    assert out.shape == (2, 100, 187)
    assert sum(p.numel() for p in model.parameters()) == 1_271_908


def test_translator_legacy_path_unchanged_without_positions():
    torch.manual_seed(0)
    model = MuQToMelTranslator(1024, 32, 100)
    x = torch.randn(1, 50, 1024)
    legacy = model(x, 187)
    h = F.interpolate(model.input_projection(x).transpose(1, 2), size=187, mode="linear", align_corners=False)
    manual = model.output_projection(model.temporal_conv(h.transpose(1, 2))).transpose(1, 2)
    torch.testing.assert_close(legacy, manual)


def test_translator_rejects_wrong_position_shape():
    model = MuQToMelTranslator(1024, 32, 100)
    with pytest.raises(ValueError):
        model(torch.randn(1, 50, 1024), 187, torch.zeros(186))


# ---- decode-time mel floor ----


@needs_bigvgan
def test_mel_floor_derived_from_official_preprocessing():
    assert bigvgan_log_mel_floor() == pytest.approx(math.log(1e-5))


def test_clamp_to_mel_floor_counts_raised_bins():
    mel = torch.tensor([[-12.0, -11.6, -11.0, 0.0]])
    clamped, n = clamp_to_mel_floor(mel, math.log(1e-5))
    assert n == 2
    assert clamped.min().item() == pytest.approx(math.log(1e-5))
    assert clamped[0, 2:].tolist() == [-11.0, 0.0]
