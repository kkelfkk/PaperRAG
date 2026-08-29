"""Deterministic information-retrieval ranking metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def _deduplicate_ranking(retrieved_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(retrieved_ids))


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    _validate_k(k)
    ranking = _deduplicate_ranking(retrieved_ids)[:k]
    relevant_retrieved = sum(chunk_id in relevant_ids for chunk_id in ranking)
    return relevant_retrieved / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    _validate_k(k)
    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty for recall")
    ranking = _deduplicate_ranking(retrieved_ids)[:k]
    relevant_retrieved = len(set(ranking).intersection(relevant_ids))
    return relevant_retrieved / len(relevant_ids)


def reciprocal_rank_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    _validate_k(k)
    for rank, chunk_id in enumerate(_deduplicate_ranking(retrieved_ids)[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _dcg(grades: Sequence[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_grades: Mapping[str, int],
    k: int,
) -> float:
    _validate_k(k)
    if not relevance_grades:
        raise ValueError("relevance_grades cannot be empty for nDCG")
    ranking = _deduplicate_ranking(retrieved_ids)[:k]
    actual = [relevance_grades.get(chunk_id, 0) for chunk_id in ranking]
    ideal = sorted(relevance_grades.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    return _dcg(actual) / ideal_dcg if ideal_dcg else 0.0


def evaluate_ranking(
    retrieved_ids: Sequence[str],
    relevance_grades: Mapping[str, int],
    cutoffs: Sequence[int],
) -> dict[str, float]:
    """Compute all baseline metrics for one ranked result list."""

    normalized_cutoffs = tuple(sorted(set(cutoffs)))
    if not normalized_cutoffs:
        raise ValueError("cutoffs cannot be empty")
    relevant_ids = set(relevance_grades)
    if not relevant_ids:
        raise ValueError("relevance_grades cannot be empty")

    metrics: dict[str, float] = {}
    for k in normalized_cutoffs:
        _validate_k(k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"mrr@{k}"] = reciprocal_rank_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevance_grades, k)
    return metrics
