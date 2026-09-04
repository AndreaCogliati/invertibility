"""Reconstruction-only losses for Gate 3+ decoder training (CLAUDE.md Decoder Design).

Both losses run on raw waveforms. Their FFT/mel configurations are chosen
explicitly for 24kHz music, not copied from a 22.05kHz-speech HiFi-GAN
recipe -- see notes/DECISIONS.md ("gate3-loss-configs"). Confirmed to run
natively on MPS (no CPU fallback needed for `torch.stft` or
`torchaudio.transforms.MelSpectrogram` in this environment).
"""

from __future__ import annotations

import torch
import torchaudio
from torch import nn


class MelSpectrogramLoss(nn.Module):
    """L1 loss between log-magnitude mel spectrograms of predicted and target waveforms."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        fmin: float,
        fmax: float,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=1.0,
        )
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: (batch, num_samples) -> scalar L1 loss on log-mel magnitude."""
        pred_mel = self.mel(pred)
        target_mel = self.mel(target)
        return nn.functional.l1_loss(
            torch.log(pred_mel.clamp_min(self.eps)), torch.log(target_mel.clamp_min(self.eps))
        )


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral-convergence + log-magnitude L1 loss averaged over several STFT resolutions."""

    def __init__(
        self,
        fft_sizes: tuple[int, ...],
        hop_sizes: tuple[int, ...],
        win_lengths: tuple[int, ...],
        eps: float = 1e-7,
    ):
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("fft_sizes, hop_sizes, win_lengths must have equal length")
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.eps = eps
        for win_length in set(win_lengths):
            self.register_buffer(
                f"window_{win_length}", torch.hann_window(win_length), persistent=False
            )

    def _magnitude(self, x: torch.Tensor, fft_size: int, hop_size: int, win_length: int) -> torch.Tensor:
        window = getattr(self, f"window_{win_length}")
        spec = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        return spec.abs().clamp_min(self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """pred, target: (batch, num_samples) -> (total, spectral_convergence, log_magnitude)."""
        sc_loss = pred.new_zeros(())
        mag_loss = pred.new_zeros(())
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pred_mag = self._magnitude(pred, fft_size, hop_size, win_length)
            target_mag = self._magnitude(target, fft_size, hop_size, win_length)
            sc_loss = sc_loss + torch.norm(target_mag - pred_mag, p="fro") / torch.norm(
                target_mag, p="fro"
            ).clamp_min(self.eps)
            mag_loss = mag_loss + nn.functional.l1_loss(torch.log(pred_mag), torch.log(target_mag))
        n = len(self.fft_sizes)
        sc_loss = sc_loss / n
        mag_loss = mag_loss / n
        return sc_loss + mag_loss, sc_loss, mag_loss
