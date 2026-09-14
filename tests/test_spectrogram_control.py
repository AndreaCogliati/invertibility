import numpy as np
import pytest
import torch

from muq_inversion.latent_diagnostics import compute_latent_stats, ratio
from muq_inversion.models.spectrogram_translator import SpectrogramToDacTranslator
from muq_inversion.spectrogram_encoder import (
    HOP_LENGTH,
    N_MELS,
    SAMPLE_RATE,
    LogMelSpectrogramEncoder,
    align_to_dac_frames,
)

# --- temporal alignment ---


def test_spectrogram_frame_rate_matches_dac_hop():
    """hop_length must equal 24000/75 = 320, DAC's own measured hop -- the whole point of this
    representation choice is landing on DAC's grid with no repeat-interleave step."""
    assert HOP_LENGTH == SAMPLE_RATE // 75


def test_align_to_dac_frames_truncates_one_frame_excess():
    spec = torch.randn(1, 151, N_MELS)
    aligned = align_to_dac_frames(spec, 150)
    assert aligned.shape == (1, 150, N_MELS)
    assert torch.equal(aligned, spec[:, :150, :])


def test_align_to_dac_frames_noop_when_already_matched():
    spec = torch.randn(1, 150, N_MELS)
    aligned = align_to_dac_frames(spec, 150)
    assert torch.equal(aligned, spec)


def test_align_to_dac_frames_rejects_large_mismatch():
    spec = torch.randn(1, 140, N_MELS)
    with pytest.raises(ValueError):
        align_to_dac_frames(spec, 150)


def test_encoder_produces_expected_frame_count_before_alignment():
    """2.0s / 48000-sample crop -> 151 frames under torchaudio's center=True padding, one more
    than DAC's 150 -- measured directly, not assumed (module docstring)."""
    encoder = LogMelSpectrogramEncoder(torch.device("cpu"))
    waveform = torch.randn(1, 48000)
    features = encoder.encode(waveform)
    assert features.shape == (1, 151, N_MELS)
    assert torch.isfinite(features).all()


# --- translator shape / parameter sharing ---


def test_translator_output_matches_dac_channels():
    model = SpectrogramToDacTranslator(input_channels=N_MELS, dac_channels=1024)
    x = torch.randn(1, 150, N_MELS)
    out = model(x)
    assert out.shape == (1, 150, 1024)


def test_translator_receptive_field_matches_temporal_conv_net():
    from muq_inversion.models.temporal_conv import TemporalConvNet

    model = SpectrogramToDacTranslator(input_channels=N_MELS, dac_channels=1024)
    bare = TemporalConvNet(channels=1024)
    assert model.receptive_field_frames == bare.receptive_field_frames


# --- latent diagnostics ---


def test_compute_latent_stats_detects_norm_collapse():
    rng = np.random.default_rng(0)
    healthy = rng.normal(size=(500, 32)).astype(np.float32) * 5.0
    collapsed = np.zeros((500, 32), dtype=np.float32)

    healthy_stats = compute_latent_stats(healthy)
    collapsed_stats = compute_latent_stats(collapsed)

    assert collapsed_stats.vector_norm_mean == 0.0
    assert collapsed_stats.channel_variance_mean == 0.0
    assert healthy_stats.vector_norm_mean > 0.0
    assert healthy_stats.channel_variance_mean > 0.0


def test_ratio_is_one_for_identical_distributions():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(500, 32)).astype(np.float32)
    stats = compute_latent_stats(data)
    r = ratio(stats, stats)
    assert r["norm_ratio"] == pytest.approx(1.0)
    assert r["variance_ratio"] == pytest.approx(1.0)


def test_ratio_reflects_collapse():
    rng = np.random.default_rng(2)
    oracle_data = rng.normal(size=(500, 32)).astype(np.float32) * 5.0
    predicted_data = oracle_data * 0.01  # near-collapsed relative to oracle
    oracle_stats = compute_latent_stats(oracle_data)
    predicted_stats = compute_latent_stats(predicted_data)
    r = ratio(predicted_stats, oracle_stats)
    assert r["norm_ratio"] < 0.1
    assert r["variance_ratio"] < 0.01


def test_compute_latent_stats_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        compute_latent_stats(np.zeros((10, 10, 10)))
