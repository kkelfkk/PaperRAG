"""Tests for cross-encoder scoring and second-stage retrieval."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from app.reranking.cross_encoder import (
    RerankingError,
    SentenceTransformersCrossEncoder,
)
from app.reranking.retriever import RerankingRetriever
from app.retrieval.filters import SearchFilters
from app.retrieval.models import SearchHit, SearchResponse


def _hit(chunk_id: str, rank: int, text: str) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=1.0 / rank,
        chunk_id=chunk_id,
        document_id="doc-1",
        source_file="paper.pdf",
        title="RAG Paper",
        page_number=rank,
        chunk_index=rank - 1,
        section="Methods",
        text=text,
    )


class FakeCandidateRetriever:
    def __init__(self, hits: Sequence[SearchHit]) -> None:
        self.hits = tuple(hits)
        self.calls: list[tuple[int, SearchFilters | None]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        self.calls.append((top_k, filters))
        return SearchResponse(
            query=query,
            collection_name="papers",
            embedding_model="hybrid:test+bm25-v1",
            hits=self.hits[:top_k],
        )


class FakeReranker:
    model_name = "test/cross-encoder"

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return self.scores


def test_reranking_reorders_candidates_and_replaces_scores() -> None:
    candidates = FakeCandidateRetriever(
        [
            _hit("first", 1, "A generic passage."),
            _hit("second", 2, "The directly relevant evidence."),
            _hit("third", 3, "Another passage."),
        ]
    )
    scorer = FakeReranker([0.1, 0.95, 0.3])
    retriever = RerankingRetriever(candidates, scorer)

    filters = SearchFilters(document_id="doc-1", page_from=2)
    response = retriever.search("relevant evidence", top_k=2, filters=filters)

    assert candidates.calls == [(20, filters)]
    assert [hit.chunk_id for hit in response.hits] == ["second", "third"]
    assert [hit.rank for hit in response.hits] == [1, 2]
    assert [hit.score for hit in response.hits] == [0.95, 0.3]
    assert response.embedding_model.endswith("|reranker:test/cross-encoder")
    query, passages = scorer.calls[0]
    assert query == "relevant evidence"
    assert "Title: RAG Paper" in passages[0]
    assert "Section: Methods" in passages[0]


def test_empty_candidates_skip_cross_encoder() -> None:
    candidates = FakeCandidateRetriever([])
    scorer = FakeReranker([])

    response = RerankingRetriever(candidates, scorer).search("question")

    assert response.hits == ()
    assert scorer.calls == []


def test_mismatched_score_count_is_rejected() -> None:
    candidates = FakeCandidateRetriever([_hit("one", 1, "text")])

    with pytest.raises(RerankingError, match="score count"):
        RerankingRetriever(candidates, FakeReranker([])).search("question")


class FakeModel:
    def __init__(self, predictions: Sequence[float]) -> None:
        self.predictions = predictions
        self.pairs: object = None

    def predict(self, pairs: object, **kwargs: object) -> Sequence[float]:
        del kwargs
        self.pairs = pairs
        return self.predictions


def test_sentence_transformers_adapter_validates_predictions() -> None:
    provider = SentenceTransformersCrossEncoder("test/model")
    model = FakeModel([0.2, 0.8])
    provider._model = model

    scores = provider.score("query", ["first", "second"])

    assert scores == [0.2, 0.8]
    assert model.pairs == [("query", "first"), ("query", "second")]

    provider._model = FakeModel([math.nan])
    with pytest.raises(RerankingError, match="non-finite"):
        provider.score("query", ["first"])
