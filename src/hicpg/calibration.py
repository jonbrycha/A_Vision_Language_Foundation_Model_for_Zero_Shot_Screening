from dataclasses import dataclass
from math import ceil
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SplitConformalCalibrator:
    alpha: float
    quantile: float
    class_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        probabilities: np.ndarray,
        labels: np.ndarray,
        class_names: Sequence[str],
        alpha: float = 0.10,
    ) -> "SplitConformalCalibrator":
        matrix = np.asarray(probabilities, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        names = tuple(class_names)
        if matrix.ndim != 2:
            raise ValueError("probabilities must be a matrix")
        if targets.ndim != 1 or targets.shape[0] != matrix.shape[0]:
            raise ValueError("labels must align with probabilities")
        if matrix.shape[1] != len(names):
            raise ValueError("class names must align with probability columns")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one")
        if matrix.shape[0] == 0:
            raise ValueError("calibration set is empty")
        if np.any(targets < 0) or np.any(targets >= matrix.shape[1]):
            raise ValueError("labels contain an invalid class index")
        nonconformity = 1.0 - matrix[np.arange(matrix.shape[0]), targets]
        rank = min(ceil((matrix.shape[0] + 1) * (1.0 - alpha)), matrix.shape[0])
        quantile = float(np.partition(nonconformity, rank - 1)[rank - 1])
        return cls(alpha=alpha, quantile=quantile, class_names=names)

    @property
    def threshold(self) -> float:
        return 1.0 - self.quantile

    def predict_mask(self, probabilities: np.ndarray) -> np.ndarray:
        matrix = np.asarray(probabilities, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.class_names):
            raise ValueError("probability dimensions do not match calibrator")
        return matrix >= self.threshold

    def predict_sets(self, probabilities: np.ndarray) -> list[tuple[str, ...]]:
        mask = self.predict_mask(probabilities)
        return [tuple(name for name, keep in zip(self.class_names, row, strict=True) if keep) for row in mask]

    def coverage(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        mask = self.predict_mask(probabilities)
        targets = np.asarray(labels, dtype=np.int64)
        if targets.shape != (mask.shape[0],):
            raise ValueError("labels must align with probabilities")
        return float(mask[np.arange(mask.shape[0]), targets].mean())

    def mean_set_size(self, probabilities: np.ndarray) -> float:
        return float(self.predict_mask(probabilities).sum(axis=1).mean())

    def to_dict(self) -> dict[str, object]:
        return {"alpha": self.alpha, "quantile": self.quantile, "class_names": list(self.class_names)}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SplitConformalCalibrator":
        alpha = float(value["alpha"])
        quantile = float(value["quantile"])
        class_names = tuple(str(item) for item in value["class_names"])
        return cls(alpha, quantile, class_names)
