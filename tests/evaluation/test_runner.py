"""Tests for the retrieval evaluation runner."""

from __future__ import annotations

from typing import ClassVar

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.runner import evaluate_retriever
from app.retrieval.filters import SearchFilters
from app.retrieval.models import SearchHit, SearchResponse


class FakeSearcher:
    rankings: ClassVar[dict[str, tuple[str, ...]]] = {
        "first question": ("a", "x"),
        "second question": ("x", "b"),
    }

    def __init__(self) -> None:
        self.filters: list[SearchFilters | None] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        self.filters.append(filters)
        hits = tuple(
            SearchHit(
                rank=rank,
                score=1.0 / rank,
                chunk_id=chunk_id,
                document_id="doc-1",
                source_file="paper.pdf",
                title="Paper",
                page_number=rank,
                chunk_index=rank - 1,
                section=None,
                text=chunk_id,
            )
            for rank, chunk_id in enumerate(self.rankings[query][:top_k], start=1)
        )
        return SearchResponse(
            query=query,
            collection_name="test-collection",
            embedding_model="test-model",
            hits=hits,
        )


def _dataset() -> EvaluationDataset:
    return EvaluationDataset.model_validate(
        {
            "name": "runner-test",
            "version": "1.0.0",
            "description": "Runner aggregation test.",
            "corpus_id": "corpus-v1",
            "queries": [
                {
                    "query_id": "q1",
                    "question": "first question",
                    "question_type": "factual",
                    "relevant_chunk_ids": ["a"],
                    "section": "Methods",
                    "page_from": 3,
                    "page_to": 8,
                },
                {
                    "query_id": "q2",
                    "question": "second question",
                    "question_type": "terminology",
                    "relevant_chunk_ids": ["b"],
                },
            ],
        }
    )


def test_runner_returns_per_query_and_macro_average_metrics() -> None:
    searcher = FakeSearcher()
    report = evaluate_retriever(_dataset(), searcher, cutoffs=[1, 2])

    assert report.query_count == 2
    assert report.collection_name == "test-collection"
    assert report.embedding_model == "test-model"
    assert report.summary["recall@2"] == 1.0
    assert report.summary["mrr@2"] == 0.75
    assert report.queries[1].retrieved_chunk_ids == ("x", "b")
    assert report.to_dict()["dataset_version"] == "1.0.0"
    assert searcher.filters[0] == SearchFilters(
        section="Methods",
        page_from=3,
        page_to=8,
    )
