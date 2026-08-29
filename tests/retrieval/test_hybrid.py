"""Tests for Reciprocal Rank Fusion and hybrid retrieval."""

from __future__ import annotations

from app.retrieval.filters import SearchFilters
from app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.models import SearchHit, SearchResponse


def _hit(chunk_id: str, rank: int) -> SearchHit:
    return SearchHit(
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


class FakeSearcher:
    def __init__(self, model: str, chunk_ids: tuple[str, ...]) -> None:
        self.model = model
        self.chunk_ids = chunk_ids
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
            embedding_model=self.model,
            hits=tuple(
                _hit(chunk_id, rank)
                for rank, chunk_id in enumerate(self.chunk_ids[:top_k], start=1)
            ),
        )


def test_rrf_promotes_result_found_by_both_retrievers() -> None:
    fused = reciprocal_rank_fusion(
        [
            [_hit("dense-only", 1), _hit("shared", 2)],
            [_hit("shared", 1), _hit("sparse-only", 2)],
        ],
        rrf_k=60,
        top_k=3,
    )

    assert [hit.chunk_id for hit in fused] == [
        "shared",
        "dense-only",
        "sparse-only",
    ]
    assert [hit.rank for hit in fused] == [1, 2, 3]
    assert fused[0].score > fused[1].score


def test_hybrid_fetches_broad_candidates_and_preserves_filter() -> None:
    dense = FakeSearcher("dense-model", ("a", "shared"))
    sparse = FakeSearcher("bm25-v1", ("shared", "b"))
    retriever = HybridRetriever(dense, sparse, candidate_multiplier=3)

    filters = SearchFilters(document_id="doc-1", section="Methods")
    response = retriever.search("question", top_k=2, filters=filters)

    assert dense.calls == [(6, filters)]
    assert sparse.calls == [(6, filters)]
    assert response.embedding_model == "hybrid:dense-model+bm25-v1"
    assert [hit.chunk_id for hit in response.hits] == ["shared", "a"]
