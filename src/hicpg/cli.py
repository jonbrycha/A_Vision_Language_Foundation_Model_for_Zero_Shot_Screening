import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from hicpg.calibration import SplitConformalCalibrator
from hicpg.config import load_config
from hicpg.data import read_manifest, stratified_patient_split, write_manifest
from hicpg.metrics import balanced_accuracy, binary_metrics, brier_score, expected_calibration_error
from hicpg.prompts import PROMPTS, validate_prompt_library


def _read_scores(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "label" not in reader.fieldnames:
            raise ValueError("score file requires a label column")
        class_names = [name.removeprefix("probability_") for name in reader.fieldnames if name.startswith("probability_")]
        rows = list(reader)
    if not class_names:
        raise ValueError("score file has no probability columns")
    labels = np.asarray([class_names.index(row["label"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray([[float(row[f"probability_{name}"]) for name in class_names] for row in rows], dtype=np.float64)
    return class_names, labels, probabilities


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def inspect_prompts(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    validate_prompt_library()
    value = {
        "classes": len(PROMPTS),
        "levels": 3,
        "templates_per_level": config.scoring.templates_per_level,
        "total_templates": sum(sum(len(level) for level in group.levels()) for group in PROMPTS.values()),
        "level_weights": config.scoring.level_weights,
        "temperature": config.scoring.temperature,
    }
    print(json.dumps(value, indent=2))


def split_manifest(args: argparse.Namespace) -> None:
    records = read_manifest(args.manifest)
    calibration, evaluation = stratified_patient_split(records, args.evaluation_fraction, args.seed)
    write_manifest(args.output / "calibration.csv", calibration)
    write_manifest(args.output / "evaluation.csv", evaluation)


def calibrate(args: argparse.Namespace) -> None:
    names, labels, probabilities = _read_scores(args.scores)
    calibrator = SplitConformalCalibrator.fit(probabilities, labels, names, args.alpha)
    _write_json(args.output, calibrator.to_dict())


def evaluate(args: argparse.Namespace) -> None:
    names, labels, probabilities = _read_scores(args.scores)
    value = json.loads(args.calibrator.read_text(encoding="utf-8"))
    calibrator = SplitConformalCalibrator.from_dict(value)
    if tuple(names) != calibrator.class_names:
        raise ValueError("score classes differ from calibrator")
    predictions = probabilities.argmax(axis=1)
    result: dict[str, object] = {
        "balanced_accuracy": balanced_accuracy(labels, predictions),
        "coverage": calibrator.coverage(probabilities, labels),
        "mean_set_size": calibrator.mean_set_size(probabilities),
        "ece": expected_calibration_error(probabilities, labels),
        "brier_score": brier_score(probabilities, labels),
    }
    if len(names) == 2:
        result["binary"] = binary_metrics(labels, probabilities[:, 1]).__dict__
    _write_json(args.output, result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hicpg")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-prompts")
    inspect.add_argument("--config", type=Path, required=True)
    inspect.set_defaults(handler=inspect_prompts)
    split = commands.add_parser("split")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--evaluation-fraction", type=float, default=0.20)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(handler=split_manifest)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--scores", type=Path, required=True)
    calibration.add_argument("--alpha", type=float, default=0.10)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.set_defaults(handler=calibrate)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--scores", type=Path, required=True)
    evaluation.add_argument("--calibrator", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.set_defaults(handler=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
