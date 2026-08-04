from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ModelConfig:
    name: str
    image_size: int
    embedding_dim: int
    frozen: bool


@dataclass(frozen=True)
class ScoringConfig:
    level_weights: tuple[float, float, float]
    templates_per_level: int
    temperature: float
    calibration_temperature: float


@dataclass(frozen=True)
class ConformalConfig:
    alpha: float
    calibration_fraction: float


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    precision: str
    batch_size: int
    workers: int
    seed: int


@dataclass(frozen=True)
class HiCPGConfig:
    model: ModelConfig
    scoring: ScoringConfig
    conformal: ConformalConfig
    runtime: RuntimeConfig


@dataclass(frozen=True)
class ManifestRecord:
    path: Path
    label: str
    dataset: str
    patient_id: str


@dataclass(frozen=True)
class ScoreRecord:
    path: str
    label: str
    dataset: str
    scores: Mapping[str, float]


@dataclass(frozen=True)
class PredictionRecord:
    path: str
    label: str
    prediction: str
    confidence: float
    prediction_set: Sequence[str]
