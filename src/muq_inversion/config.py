"""Versioned YAML configuration schema.

All tunable parameters live here and in the YAML files under configs/.
Nothing important should be hard-coded as a default deep in the code.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class EnvironmentConfig:
    seed: int
    device: str = "auto"  # one of: auto, mps, cpu
    allow_cpu_fallback: bool = False
    dtype: str = "float32"


@dataclasses.dataclass(frozen=True)
class MuQConfig:
    model_id: str = "OpenMuQ/MuQ-large-msd-iter"
    sample_rate: int = 24000


@dataclasses.dataclass(frozen=True)
class EncodecConfig:
    """EnCodec representation-source experiment -- see notes/DECISIONS.md#encodec-hifigan.
    Uses the continuous pre-quantization encoder embedding (`model.encoder(...)`), not the
    quantized RVQ codes, to stay analogous to MuQ's continuous last_hidden_state."""

    model_id: str = "facebook/encodec_24khz"
    sample_rate: int = 24000


@dataclasses.dataclass(frozen=True)
class DataConfig:
    raw_dir: str = "data/raw/gtzan"
    processed_dir: str = "data/processed"
    manifests_dir: str = "data/manifests"
    source: str = "hf:marsyas/gtzan"
    expected_track_count: int = 1000
    genres: tuple[str, ...] = (
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    )
    original_sample_rate: int = 22050
    target_sample_rate: int = 24000
    split_seed: int = 1234
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1


@dataclasses.dataclass(frozen=True)
class CacheConfig:
    cache_dir: str = "data/cache"
    dtype: str = "float32"  # one of: float32, float16
    representation_name: str = "last_hidden_state"


@dataclasses.dataclass(frozen=True)
class FeatureContractConfig:
    """Gate 2 validation parameters."""

    probe_durations_seconds: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
    probe_track_count: int = 3
    frame_rate_tolerance_hz: float = 0.5
    alignment_tolerance_frames: int = 1
    lowprecision_dtype: str = "float16"
    lowprecision_max_abs_error: float = 0.05


@dataclasses.dataclass(frozen=True)
class DecoderConfig:
    """upsample_rates' product (960) was confirmed against real GTZAN audio in Gate 2
    (see notes/DECISIONS.md#decoder-upsample-placeholder): MuQ downsamples by an exact
    integer factor of 960, so training crops must be exact multiples of 960 samples.

    Factored into 4 stages (8,6,5,4) rather than the original 6 (5,4,4,3,2,2): each stage halves
    the channel count, so fewer stages keep more capacity at the fine-time-resolution end of the
    network -- see notes/DECISIONS.md#gate3-generator-capacity. resblock_kernel_sizes/
    resblock_dilation_sizes configure the multi-receptive-field (MRF) fusion block run after each
    upsampling stage (several residual branches with different kernel sizes, averaged)."""

    base_channels: int = 512
    upsample_rates: tuple[int, ...] = (8, 6, 5, 4)  # product 960, 4 stages
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))


@dataclasses.dataclass(frozen=True)
class MelLossConfig:
    """Explicit for 24kHz music -- see notes/DECISIONS.md#gate3-loss-configs."""

    n_fft: int = 2048
    hop_length: int = 300
    win_length: int = 1200
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 12000.0  # Nyquist at 24kHz


@dataclasses.dataclass(frozen=True)
class STFTLossConfig:
    """Multi-resolution STFT loss configuration -- see notes/DECISIONS.md#gate3-loss-configs."""

    fft_sizes: tuple[int, ...] = (512, 1024, 2048)
    hop_sizes: tuple[int, ...] = (50, 120, 240)
    win_lengths: tuple[int, ...] = (240, 600, 1200)


@dataclasses.dataclass(frozen=True)
class DiscriminatorConfig:
    """Multi-period/multi-scale discriminators -- see notes/DECISIONS.md#gate3-adversarial-losses.
    Channel widths kept smaller than the original (speech) HiFi-GAN's 1024-wide discriminators,
    consistent with this project's "deliberately small" baseline."""

    period_periods: tuple[int, ...] = (2, 3, 5, 7, 11)
    period_channels: tuple[int, ...] = (32, 64, 128, 256, 256)
    period_kernel_size: int = 5
    period_stride: int = 3
    scale_channels: tuple[int, ...] = (32, 64, 128, 256, 256)
    num_scales: int = 3


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Gate 3 (tiny-set overfit) training loop parameters."""

    overfit_track_ids: tuple[str, ...] = ()  # empty: pick num_overfit_tracks deterministically
    num_overfit_tracks: int = 2
    crop_frames: int = 50  # crop_samples = crop_frames * decoder.total_upsample
    steps: int = 500
    learning_rate: float = 2.0e-4
    log_every: int = 50
    checkpoint_every: int = 250
    loss_decrease_ratio: float = 0.5  # PASS if final/initial loss <= this
    run_name: str = "gate3_tiny_overfit"
    # Adversarial training (CLAUDE.md Decoder Design item 6): reconstruction-only for the first
    # adversarial_warmup_steps, then discriminators are added -- only after the reconstruction
    # path has already been shown to converge (see notes/DECISIONS.md#gate3-adversarial-losses).
    adversarial_warmup_steps: int = 500
    discriminator_learning_rate: float = 2.0e-4
    adversarial_loss_weight: float = 1.0
    feature_matching_loss_weight: float = 2.0
    # Applied to (mel + STFT) during the adversarial phase only, so the reconstruction term stays
    # the dominant driver of *content* fidelity (pitch/melody) while adversarial+feature-matching
    # refine perceptual naturalness (phase) on top -- not a bare copy of HiFi-GAN's "45" constant,
    # since our loss combination differs (we also have a multi-res STFT term); chosen from the
    # measured loss magnitudes in the first adversarial run. See
    # notes/DECISIONS.md#gate3-adversarial-losses.
    reconstruction_loss_weight: float = 10.0
    # Adam betas for the generator/discriminator optimizers. PyTorch's default (0.9, 0.999) is a
    # known cause of discriminator-overpowers-generator collapse in adversarial training (too much
    # momentum in a minimax game); (0.8, 0.99) matches the original HiFi-GAN paper's published
    # config. See notes/DECISIONS.md#gate3-capacity-1024 for the collapse this was diagnosed from.
    generator_betas: tuple[float, float] = (0.8, 0.99)
    discriminator_betas: tuple[float, float] = (0.8, 0.99)


@dataclasses.dataclass(frozen=True)
class PilotConfig:
    """Gate 4 (short pilot) parameters: a larger deterministic subset than Gate 3, plus
    checkpoint/resume equivalence and conditioning-control settings."""

    track_ids: tuple[str, ...] = ()  # empty: pick num_tracks deterministically
    num_tracks: int = 6
    crop_frames: int = 50  # crop_samples = crop_frames * decoder.total_upsample
    steps: int = 200
    learning_rate: float = 2.0e-4
    log_every: int = 50
    checkpoint_every: int = 100
    resume_test_at: int = 100  # step at which to checkpoint, tear down, and resume
    resume_tolerance: float = 0.02  # max relative loss difference, resumed vs uninterrupted
    control_margin: float = 0.2  # correct-conditioning loss must be <= (1 - margin) * control loss
    run_name: str = "gate4_short_pilot"


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    name: str
    environment: EnvironmentConfig
    muq: MuQConfig
    data: DataConfig
    cache: CacheConfig
    feature_contract: FeatureContractConfig
    decoder: DecoderConfig
    mel_loss: MelLossConfig
    stft_loss: STFTLossConfig
    discriminator: DiscriminatorConfig
    training: TrainingConfig
    pilot: PilotConfig
    # Representation-source selector for the encodec-hifigan-experiment branch (see
    # notes/DECISIONS.md#encodec-hifigan). Defaults to "muq" so every existing config's resolved
    # behavior is unchanged unless a config explicitly opts into "encodec".
    encoder_type: str = "muq"
    encodec: EncodecConfig = dataclasses.field(default_factory=EncodecConfig)


def _dataclass_from_dict(cls: type, payload: dict[str, Any]) -> Any:
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - field_names
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**payload)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    environment = _dataclass_from_dict(EnvironmentConfig, raw.get("environment", {}))
    muq = _dataclass_from_dict(MuQConfig, raw.get("muq", {}))
    data_raw = dict(raw.get("data", {}))
    if "genres" in data_raw:
        data_raw["genres"] = tuple(data_raw["genres"])
    data = _dataclass_from_dict(DataConfig, data_raw)

    cache = _dataclass_from_dict(CacheConfig, raw.get("cache", {}))

    feature_contract_raw = dict(raw.get("feature_contract", {}))
    if "probe_durations_seconds" in feature_contract_raw:
        feature_contract_raw["probe_durations_seconds"] = tuple(
            feature_contract_raw["probe_durations_seconds"]
        )
    feature_contract = _dataclass_from_dict(FeatureContractConfig, feature_contract_raw)

    decoder_raw = dict(raw.get("decoder", {}))
    if "upsample_rates" in decoder_raw:
        decoder_raw["upsample_rates"] = tuple(decoder_raw["upsample_rates"])
    if "resblock_kernel_sizes" in decoder_raw:
        decoder_raw["resblock_kernel_sizes"] = tuple(decoder_raw["resblock_kernel_sizes"])
    if "resblock_dilation_sizes" in decoder_raw:
        decoder_raw["resblock_dilation_sizes"] = tuple(
            tuple(d) for d in decoder_raw["resblock_dilation_sizes"]
        )
    decoder = _dataclass_from_dict(DecoderConfig, decoder_raw)

    mel_loss = _dataclass_from_dict(MelLossConfig, raw.get("mel_loss", {}))

    stft_loss_raw = dict(raw.get("stft_loss", {}))
    for key in ("fft_sizes", "hop_sizes", "win_lengths"):
        if key in stft_loss_raw:
            stft_loss_raw[key] = tuple(stft_loss_raw[key])
    stft_loss = _dataclass_from_dict(STFTLossConfig, stft_loss_raw)

    discriminator_raw = dict(raw.get("discriminator", {}))
    for key in ("period_periods", "period_channels", "scale_channels"):
        if key in discriminator_raw:
            discriminator_raw[key] = tuple(discriminator_raw[key])
    discriminator = _dataclass_from_dict(DiscriminatorConfig, discriminator_raw)

    training_raw = dict(raw.get("training", {}))
    if "overfit_track_ids" in training_raw:
        training_raw["overfit_track_ids"] = tuple(training_raw["overfit_track_ids"])
    for key in ("generator_betas", "discriminator_betas"):
        if key in training_raw:
            training_raw[key] = tuple(training_raw[key])
    training = _dataclass_from_dict(TrainingConfig, training_raw)

    pilot_raw = dict(raw.get("pilot", {}))
    if "track_ids" in pilot_raw:
        pilot_raw["track_ids"] = tuple(pilot_raw["track_ids"])
    pilot = _dataclass_from_dict(PilotConfig, pilot_raw)

    encodec = _dataclass_from_dict(EncodecConfig, raw.get("encodec", {}))

    return ExperimentConfig(
        name=raw["name"],
        environment=environment,
        muq=muq,
        data=data,
        cache=cache,
        feature_contract=feature_contract,
        decoder=decoder,
        mel_loss=mel_loss,
        stft_loss=stft_loss,
        discriminator=discriminator,
        training=training,
        pilot=pilot,
        encoder_type=raw.get("encoder_type", "muq"),
        encodec=encodec,
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)
