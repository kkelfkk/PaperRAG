"""Tests for deterministic information-retrieval metrics."""

from __future__ import annotations

import math

import pytest

from app.evaluation.metrics import evaluate_ranking, recall_at_k


def test_evaluate_ranking_computes_expected_values() -> None:
    metrics = evaluate_ranking(
        ["high", "irrelevant", "low"],
        {"high": 2, "low": 1},
        [1, 3],
    )

    expected_ndcg_at_3 = (3 + 1 / math.log2(4)) / (3 + 1 / math.log2(3))
    assert metrics["precision@1"] == 1.0
    assert metrics["recall@1"] == 0.5
    assert metrics["mrr@3"] == 1.0
    assert metrics["ndcg@3"] == pytest.approx(expected_ndcg_at_3)


def test_duplicate_results_are_counted_only_once() -> None:
    metrics = evaluate_ranking(["a", "a", "b"], {"a": 1, "b": 1}, [2])

    assert metrics["precision@2"] == 1.0
    assert metrics["recall@2"] == 1.0


def test_invalid_metric_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="cutoffs cannot be empty"):
        evaluate_ranking(["a"], {"a": 1}, [])
    with pytest.raises(ValueError, match="positive"):
        evaluate_ranking(["a"], {"a": 1}, [0])
    with pytest.raises(ValueError, match="cannot be empty"):
        recall_at_k(["a"], set(), 1)
