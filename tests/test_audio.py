import numpy as np
import pytest

from muq_inversion.audio import AudioValidationError, resample, validate_waveform


def test_resample_identity_when_rates_match():
    waveform = np.random.randn(1000).astype(np.float32)
    out = resample(waveform, 24000, 24000)
    np.testing.assert_array_equal(out, waveform)


def test_resample_length_matches_ratio():
    orig_sr, target_sr = 22050, 24000
    duration_seconds = 3.0
    waveform = np.random.randn(int(orig_sr * duration_seconds)).astype(np.float32)
    out = resample(waveform, orig_sr, target_sr)
    expected_len = round(len(waveform) * target_sr / orig_sr)
    assert abs(len(out) - expected_len) <= 1


def test_validate_waveform_rejects_empty():
    with pytest.raises(AudioValidationError):
        validate_waveform(np.array([], dtype=np.float32), source_id="test")


def test_validate_waveform_rejects_nan():
    waveform = np.array([0.1, float("nan"), 0.2], dtype=np.float32)
    with pytest.raises(AudioValidationError):
        validate_waveform(waveform, source_id="test")


def test_validate_waveform_rejects_inf():
    waveform = np.array([0.1, float("inf"), 0.2], dtype=np.float32)
    with pytest.raises(AudioValidationError):
        validate_waveform(waveform, source_id="test")


def test_validate_waveform_rejects_silence():
    waveform = np.zeros(1000, dtype=np.float32)
    with pytest.raises(AudioValidationError):
        validate_waveform(waveform, source_id="test")


def test_validate_waveform_rejects_mostly_clipped():
    waveform = np.full(1000, 0.9999, dtype=np.float32)
    with pytest.raises(AudioValidationError):
        validate_waveform(waveform, source_id="test")


def test_validate_waveform_accepts_normal_signal():
    t = np.linspace(0, 1, 24000, dtype=np.float32)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    validate_waveform(waveform, source_id="test")
