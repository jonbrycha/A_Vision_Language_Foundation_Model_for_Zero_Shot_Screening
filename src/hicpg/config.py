from pathlib import Path
from typing import Any

import yaml

from hicpg.schema import ConformalConfig, HiCPGConfig, ModelConfig, RuntimeConfig, ScoringConfig


def load_config(path: Path) -> HiCPGConfig:
    value: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = value["model"]
    scoring = value["scoring"]
    conformal = value["conformal"]
    runtime = value["runtime"]
    weights = tuple(float(item) for item in scoring["level_weights"])
    if len(weights) != 3:
        raise ValueError("three prompt level weights are required")
    return HiCPGConfig(
        model=ModelConfig(str(model["name"]), int(model["image_size"]), int(model["embedding_dim"]), bool(model["frozen"])),
        scoring=ScoringConfig((weights[0], weights[1], weights[2]), int(scoring["templates_per_level"]), float(scoring["temperature"]), float(scoring["calibration_temperature"])),
        conformal=ConformalConfig(float(conformal["alpha"]), float(conformal["calibration_fraction"])),
        runtime=RuntimeConfig(str(runtime["device"]), str(runtime["precision"]), int(runtime["batch_size"]), int(runtime["workers"]), int(runtime["seed"])),
    )
