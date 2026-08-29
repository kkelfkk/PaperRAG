"""Retrieval evaluation datasets, metrics, and runners."""

from app.evaluation.dataset import EvaluationDataset, EvaluationQuery, load_dataset
from app.evaluation.metrics import evaluate_ranking
from app.evaluation.runner import (
    QueryEvaluation,
    RetrievalEvaluationReport,
    evaluate_retriever,
)

__all__ = [
    "EvaluationDataset",
    "EvaluationQuery",
    "QueryEvaluation",
    "RetrievalEvaluationReport",
    "evaluate_ranking",
    "evaluate_retriever",
    "load_dataset",
]
