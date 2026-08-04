from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class HierarchicalScores:
    template_scores: Tensor
    level_scores: Tensor
    class_scores: Tensor
    probabilities: Tensor


def cosine_template_scores(image_embeddings: Tensor, text_embeddings: Tensor) -> Tensor:
    if image_embeddings.ndim != 2:
        raise ValueError("image embeddings must have shape batch by dimension")
    if text_embeddings.ndim != 4:
        raise ValueError("text embeddings must have shape classes by levels by templates by dimension")
    if image_embeddings.shape[-1] != text_embeddings.shape[-1]:
        raise ValueError("embedding dimensions differ")
    images = F.normalize(image_embeddings.float(), dim=-1)
    texts = F.normalize(text_embeddings.float(), dim=-1)
    return torch.einsum("bd,clkd->bclk", images, texts)


def aggregate_scores(template_scores: Tensor, weights: Sequence[float]) -> tuple[Tensor, Tensor]:
    if template_scores.ndim != 4:
        raise ValueError("template scores must have four dimensions")
    if len(weights) != template_scores.shape[2]:
        raise ValueError("weight count differs from prompt level count")
    weight_tensor = torch.as_tensor(weights, dtype=template_scores.dtype, device=template_scores.device)
    if torch.any(weight_tensor < 0):
        raise ValueError("weights must be nonnegative")
    if not torch.isclose(weight_tensor.sum(), torch.ones((), device=weight_tensor.device), atol=1e-6):
        raise ValueError("weights must sum to one")
    level_scores = template_scores.mean(dim=-1)
    class_scores = torch.einsum("bcl,l->bc", level_scores, weight_tensor)
    return level_scores, class_scores


def score_embeddings(
    image_embeddings: Tensor,
    text_embeddings: Tensor,
    weights: Sequence[float],
    temperature: float,
) -> HierarchicalScores:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    template_scores = cosine_template_scores(image_embeddings, text_embeddings)
    level_scores, class_scores = aggregate_scores(template_scores, weights)
    probabilities = torch.softmax(class_scores / temperature, dim=-1)
    return HierarchicalScores(template_scores, level_scores, class_scores, probabilities)


def inverse_variance_weights(level_scores: Tensor, labels: Tensor) -> Tensor:
    if level_scores.ndim != 3:
        raise ValueError("level scores must have shape batch by classes by levels")
    if labels.ndim != 1 or labels.shape[0] != level_scores.shape[0]:
        raise ValueError("labels must align with batch")
    classes = level_scores.shape[1]
    estimates: list[Tensor] = []
    for class_index in range(classes):
        negatives = level_scores[labels != class_index, class_index]
        if negatives.shape[0] < 2:
            raise ValueError("each class requires at least two negative examples")
        estimates.append(negatives.var(dim=0, unbiased=True))
    mean_variances = torch.stack(estimates).mean(dim=0).clamp_min(torch.finfo(level_scores.dtype).eps)
    inverse = mean_variances.reciprocal()
    return inverse / inverse.sum()


def grid_search_weights(
    level_scores: Tensor,
    labels: Tensor,
    candidates: Sequence[Sequence[float]],
) -> tuple[Tensor, float]:
    from hicpg.metrics import balanced_accuracy

    best_weights: Tensor | None = None
    best_score = float("-inf")
    for candidate in candidates:
        weight_tensor = torch.as_tensor(candidate, dtype=level_scores.dtype, device=level_scores.device)
        if weight_tensor.numel() != level_scores.shape[-1]:
            raise ValueError("candidate dimension differs from prompt levels")
        if torch.any(weight_tensor < 0) or not torch.isclose(weight_tensor.sum(), torch.tensor(1.0, device=weight_tensor.device)):
            raise ValueError("candidate weights must lie on the probability simplex")
        scores = torch.einsum("bcl,l->bc", level_scores, weight_tensor)
        value = balanced_accuracy(labels.cpu().numpy(), scores.argmax(dim=-1).cpu().numpy())
        if value > best_score:
            best_score = value
            best_weights = weight_tensor
    if best_weights is None:
        raise ValueError("weight candidate list is empty")
    return best_weights, best_score
