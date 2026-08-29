"""Run a labeled dataset against any compatible retrieval implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.metrics import evaluate_ranking
from app.retrieval.models import SearchResponse


class Searcher(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> SearchResponse: ...


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    query_id: str
    question: str
    question_type: str
    relevant_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    dataset_name: str
    dataset_version: str
    corpus_id: str
    collection_name: str
    embedding_model: str
    query_count: int
    cutoffs: tuple[int, ...]
    summary: dict[str, float]
    queries: tuple[QueryEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean_metrics(results: Sequence[QueryEvaluation]) -> dict[str, float]:
    metric_names = tuple(results[0].metrics)
    return {
        name: sum(result.metrics[name] for result in results) / len(results)
        for name in metric_names
    }


def evaluate_retriever(
    dataset: EvaluationDataset,
    searcher: Searcher,
    *,
    cutoffs: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalEvaluationReport:
    """Evaluate retrieval and return per-query plus macro-average metrics."""

    normalized_cutoffs = tuple(sorted(set(cutoffs)))
    if not normalized_cutoffs or normalized_cutoffs[0] <= 0:
        raise ValueError("cutoffs must contain positive integers")
    max_k = normalized_cutoffs[-1]
    results: list[QueryEvaluation] = []
    collection_name = ""
    embedding_model = ""

    for query in dataset.queries:
        response = searcher.search(
            query.question,
            top_k=max_k,
            document_id=query.document_id,
        )
        if not collection_name:
            collection_name = response.collection_name
            embedding_model = response.embedding_model
        elif (
            response.collection_name != collection_name
            or response.embedding_model != embedding_model
        ):
            raise ValueError("retriever configuration changed during evaluation")

        retrieved = tuple(hit.chunk_id for hit in response.hits)
        results.append(
            QueryEvaluation(
                query_id=query.query_id,
                question=query.question,
                question_type=query.question_type.value,
                relevant_chunk_ids=tuple(query.relevant_chunk_ids),
                retrieved_chunk_ids=retrieved,
                metrics=evaluate_ranking(
                    retrieved,
                    query.grades(),
                    normalized_cutoffs,
                ),
            )
        )

    return RetrievalEvaluationReport(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        corpus_id=dataset.corpus_id,
        collection_name=collection_name,
        embedding_model=embedding_model,
        query_count=len(results),
        cutoffs=normalized_cutoffs,
        summary=_mean_metrics(results),
        queries=tuple(results),
    )
