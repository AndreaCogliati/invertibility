import random

import torch

from muq_inversion.pilot import _build_conditioning, _tail_mean
from muq_inversion.train import OverfitClip


class _StubEncoder:
    """Returns a fixed-shape tensor whose values encode which waveform samples were seen,
    so tests can tell which crop/clip was actually used for conditioning."""

    def __init__(self, num_frames: int = 5, channels: int = 4):
        self.num_frames = num_frames
        self.channels = channels

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        # Encode the mean of the waveform into every frame/channel, distinctly per call,
        # so shuffling frames of a constant-valued tensor is a trivial (but checkable) no-op,
        # while different source waveforms produce different constant values.
        value = waveform.mean()
        return value * torch.arange(1, self.num_frames + 1).float().view(1, -1, 1).expand(
            1, self.num_frames, self.channels
        ).clone()


def _clip(track_id: str, fill_value: float, length: int = 10_000) -> OverfitClip:
    return OverfitClip(track_id=track_id, waveform=torch.full((length,), fill_value))


def test_tail_mean_uses_last_n():
    assert _tail_mean([1.0, 2.0, 3.0, 4.0, 5.0], n=2) == 4.5


def test_tail_mean_handles_short_history():
    assert _tail_mean([1.0, 3.0], n=10) == 2.0


def test_build_conditioning_constant_zero_is_all_zero():
    encoder = _StubEncoder()
    target = _clip("a", 1.0)
    target_crop = target.waveform[:100].unsqueeze(0)
    features = _build_conditioning(
        "constant_zero", encoder, target, target_crop, [target], 100, random.Random(0)
    )
    assert torch.equal(features, torch.zeros_like(features))


def test_build_conditioning_shuffled_time_permutes_frames():
    encoder = _StubEncoder(num_frames=5, channels=2)
    target = _clip("a", 1.0)
    target_crop = target.waveform[:100].unsqueeze(0)
    features = _build_conditioning(
        "shuffled_time", encoder, target, target_crop, [target], 100, random.Random(0)
    )
    original = encoder.encode(target_crop)
    # Same multiset of per-frame values, but not guaranteed to be in the same order.
    assert sorted(features[0, :, 0].tolist()) == sorted(original[0, :, 0].tolist())


def test_build_conditioning_shuffled_across_uses_a_different_clip():
    encoder = _StubEncoder()
    target = _clip("target", fill_value=1.0)
    other = _clip("other", fill_value=99.0)
    target_crop = target.waveform[:100].unsqueeze(0)

    features = _build_conditioning(
        "shuffled_across", encoder, target, target_crop, [target, other], 100, random.Random(0)
    )
    # With only one non-target candidate, shuffled_across must always pick `other`, whose
    # encoded value (99 * frame_index) is distinguishable from the target's (1 * frame_index).
    assert features[0, 0, 0].item() == 99.0
