import torch

from muq_inversion.models.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from muq_inversion.models.losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)

BATCH = 2
NUM_SAMPLES = 4800  # multiple of every default period (2,3,5,7,11) and of 4 (MSD downsampling)


def test_multi_period_discriminator_output_shapes():
    mpd = MultiPeriodDiscriminator(
        periods=(2, 3, 5), channels=(8, 16, 16), kernel_size=5, stride=3
    )
    real = torch.randn(BATCH, NUM_SAMPLES)
    fake = torch.randn(BATCH, NUM_SAMPLES)
    real_outputs, fake_outputs, real_fmaps, _fake_fmaps = mpd(real, fake)
    assert len(real_outputs) == len(fake_outputs) == 3
    for out in real_outputs + fake_outputs:
        assert out.shape[0] == BATCH
    assert len(real_fmaps) == 3
    assert all(len(fmap) == 4 for fmap in real_fmaps)  # 3 conv layers + output_conv


def test_multi_scale_discriminator_output_shapes():
    msd = MultiScaleDiscriminator(num_scales=3, channels=(8, 16, 16))
    real = torch.randn(BATCH, NUM_SAMPLES)
    fake = torch.randn(BATCH, NUM_SAMPLES)
    real_outputs, fake_outputs, real_fmaps, _fake_fmaps = msd(real, fake)
    assert len(real_outputs) == len(fake_outputs) == 3
    for out in real_outputs + fake_outputs:
        assert out.shape[0] == BATCH
    assert len(real_fmaps) == 3


def test_scale_discriminator_rejects_invalid_groups():
    import pytest

    from muq_inversion.models.discriminators import ScaleDiscriminator

    with pytest.raises(ValueError):
        ScaleDiscriminator(channels=(7, 15))  # 7 is not divisible by group=1 fine, but 15%4!=0


def test_discriminator_loss_zero_when_real_is_one_and_fake_is_zero():
    real_outputs = [torch.ones(BATCH, 5)]
    fake_outputs = [torch.zeros(BATCH, 5)]
    loss = discriminator_loss(real_outputs, fake_outputs)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_discriminator_loss_positive_when_wrong():
    real_outputs = [torch.zeros(BATCH, 5)]
    fake_outputs = [torch.ones(BATCH, 5)]
    loss = discriminator_loss(real_outputs, fake_outputs)
    assert loss.item() > 0


def test_generator_adversarial_loss_zero_when_fake_is_one():
    fake_outputs = [torch.ones(BATCH, 5)]
    loss = generator_adversarial_loss(fake_outputs)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_feature_matching_loss_zero_for_identical_fmaps():
    fmap = [torch.randn(BATCH, 4, 10)]
    real_fmaps = [fmap]
    fake_fmaps = [fmap]
    loss = feature_matching_loss(real_fmaps, fake_fmaps)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_feature_matching_loss_positive_for_different_fmaps():
    real_fmaps = [[torch.zeros(BATCH, 4, 10)]]
    fake_fmaps = [[torch.ones(BATCH, 4, 10)]]
    loss = feature_matching_loss(real_fmaps, fake_fmaps)
    assert loss.item() > 0


def test_mpd_and_msd_forward_backward_through_generator_output():
    # Simulates the generator-step gradient path: fake requires grad, real does not.
    mpd = MultiPeriodDiscriminator(periods=(2, 3), channels=(8, 8), kernel_size=5, stride=3)
    real = torch.randn(BATCH, NUM_SAMPLES)
    fake = torch.randn(BATCH, NUM_SAMPLES, requires_grad=True)
    _real_out, fake_out, _real_fmap, _fake_fmap = mpd(real, fake)
    loss = generator_adversarial_loss(fake_out)
    loss.backward()
    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()
