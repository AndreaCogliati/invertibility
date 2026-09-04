import torch

from muq_inversion.models.losses import MelSpectrogramLoss, MultiResolutionSTFTLoss


def test_mel_loss_zero_for_identical_signals():
    torch.manual_seed(0)
    waveform = torch.randn(1, 24000)
    loss_fn = MelSpectrogramLoss(
        sample_rate=24000, n_fft=1024, hop_length=256, win_length=1024, n_mels=64, fmin=0.0, fmax=12000.0
    )
    loss = loss_fn(waveform, waveform)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_mel_loss_positive_for_different_signals():
    torch.manual_seed(0)
    a = torch.randn(1, 24000)
    b = torch.randn(1, 24000)
    loss_fn = MelSpectrogramLoss(
        sample_rate=24000, n_fft=1024, hop_length=256, win_length=1024, n_mels=64, fmin=0.0, fmax=12000.0
    )
    loss = loss_fn(a, b)
    assert loss.item() > 0


def test_stft_loss_zero_for_identical_signals():
    torch.manual_seed(0)
    waveform = torch.randn(1, 24000)
    loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=(512, 1024), hop_sizes=(128, 256), win_lengths=(512, 1024)
    )
    total, sc, mag = loss_fn(waveform, waveform)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-4)
    assert torch.allclose(sc, torch.zeros_like(sc), atol=1e-4)
    assert torch.allclose(mag, torch.zeros_like(mag), atol=1e-4)


def test_stft_loss_positive_for_different_signals():
    torch.manual_seed(0)
    a = torch.randn(1, 24000)
    b = torch.randn(1, 24000)
    loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=(512, 1024), hop_sizes=(128, 256), win_lengths=(512, 1024)
    )
    total, _sc, _mag = loss_fn(a, b)
    assert total.item() > 0


def test_stft_loss_rejects_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError):
        MultiResolutionSTFTLoss(fft_sizes=(512, 1024), hop_sizes=(128,), win_lengths=(512, 1024))
