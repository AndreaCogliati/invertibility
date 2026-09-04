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
class DecoderConfig:
    """Shape-only placeholder for Gate 0. upsample_rates is provisional until
    Gate 2 measures MuQ's actual samples-per-frame ratio (CLAUDE.md Decoder Design)."""

    base_channels: int = 256
    upsample_rates: tuple[int, ...] = (5, 4, 4, 3, 2, 2)  # product 960, provisional


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    name: str
    environment: EnvironmentConfig
    muq: MuQConfig
    data: DataConfig
    decoder: DecoderConfig


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

    decoder_raw = dict(raw.get("decoder", {}))
    if "upsample_rates" in decoder_raw:
        decoder_raw["upsample_rates"] = tuple(decoder_raw["upsample_rates"])
    decoder = _dataclass_from_dict(DecoderConfig, decoder_raw)

    return ExperimentConfig(
        name=raw["name"],
        environment=environment,
        muq=muq,
        data=data,
        decoder=decoder,
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)
