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
    integer factor of 960, so training crops must be exact multiples of 960 samples."""

    base_channels: int = 256
    upsample_rates: tuple[int, ...] = (5, 4, 4, 3, 2, 2)  # product 960


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
    training: TrainingConfig


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
    decoder = _dataclass_from_dict(DecoderConfig, decoder_raw)

    mel_loss = _dataclass_from_dict(MelLossConfig, raw.get("mel_loss", {}))

    stft_loss_raw = dict(raw.get("stft_loss", {}))
    for key in ("fft_sizes", "hop_sizes", "win_lengths"):
        if key in stft_loss_raw:
            stft_loss_raw[key] = tuple(stft_loss_raw[key])
    stft_loss = _dataclass_from_dict(STFTLossConfig, stft_loss_raw)

    training_raw = dict(raw.get("training", {}))
    if "overfit_track_ids" in training_raw:
        training_raw["overfit_track_ids"] = tuple(training_raw["overfit_track_ids"])
    training = _dataclass_from_dict(TrainingConfig, training_raw)

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
        training=training,
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)
