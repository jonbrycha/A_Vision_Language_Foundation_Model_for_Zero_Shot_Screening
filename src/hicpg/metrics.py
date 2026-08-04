from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


@dataclass(frozen=True)
class BinaryMetrics:
    auroc: float
    sensitivity: float
    specificity: float
    threshold: float


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    targets = np.asarray(labels)
    outputs = np.asarray(predictions)
    if targets.shape != outputs.shape or targets.ndim != 1:
        raise ValueError("labels and predictions must be aligned vectors")
    matrix = confusion_matrix(targets, outputs)
    support = matrix.sum(axis=1)
    recalls = np.divide(np.diag(matrix), support, out=np.zeros_like(support, dtype=float), where=support != 0)
    return float(recalls.mean())


def binary_metrics(labels: np.ndarray, positive_scores: np.ndarray) -> BinaryMetrics:
    targets = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(positive_scores, dtype=np.float64)
    if targets.shape != scores.shape or targets.ndim != 1:
        raise ValueError("labels and scores must be aligned vectors")
    fpr, tpr, thresholds = roc_curve(targets, scores)
    index = int(np.argmax(tpr - fpr))
    return BinaryMetrics(float(roc_auc_score(targets, scores)), float(tpr[index]), float(1 - fpr[index]), float(thresholds[index]))


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    matrix = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    confidence = matrix.max(axis=1)
    correct = matrix.argmax(axis=1) == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = matrix.shape[0]
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        count = int(selected.sum())
        if count:
            value += count / total * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return value


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    matrix = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    one_hot = np.eye(matrix.shape[1], dtype=np.float64)[targets]
    return float(np.square(matrix - one_hot).sum(axis=1).mean())


def bootstrap_interval(
    labels: np.ndarray,
    predictions: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    targets = np.asarray(labels)
    outputs = np.asarray(predictions)
    rng = np.random.default_rng(seed)
    classes = np.unique(targets)
    indices = {item: np.flatnonzero(targets == item) for item in classes}
    values = np.empty(resamples, dtype=np.float64)
    for iteration in range(resamples):
        sample = np.concatenate([rng.choice(group, size=group.size, replace=True) for group in indices.values()])
        values[iteration] = statistic(targets[sample], outputs[sample])
    tail = (1.0 - confidence) / 2.0
    return float(statistic(targets, outputs)), float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def bonferroni(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    return np.minimum(values * values.size, 1.0)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * values.size / np.arange(1, values.size + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def cohens_d(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    pooled = np.sqrt(((left.size - 1) * left.var(ddof=1) + (right.size - 1) * right.var(ddof=1)) / (left.size + right.size - 2))
    return float((left.mean() - right.mean()) / pooled)


def paired_bootstrap_pvalue(first: np.ndarray, second: np.ndarray) -> float:
    differences = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    if differences.ndim != 1 or differences.size < 2:
        raise ValueError("paired samples require at least two values")
    return float(stats.ttest_1samp(differences, 0.0).pvalue)
