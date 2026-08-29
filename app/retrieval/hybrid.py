"""Hybrid dense and BM25 retrieval using Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from app.retrieval.dense import DenseRetrievalError
from app.retrieval.filters import SearchFilters
from app.retrieval.models import SearchHit, SearchResponse


class Searcher(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResponse: ...


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]],
    *,
    weights: Sequence[float] | None = None,
    rrf_k: int = 60,
    top_k: int = 5,
) -> tuple[SearchHit, ...]:
    """Fuse ranked hits without requiring comparable raw retrieval scores."""

    if rrf_k < 0:
        raise DenseRetrievalError("rrf_k cannot be negative")
    if top_k <= 0:
        raise DenseRetrievalError("top_k must be positive")
    normalized_weights = tuple(weights or (1.0,) * len(rankings))
    if len(normalized_weights) != len(rankings):
        raise DenseRetrievalError("one fusion weight is required per ranking")
    if any(weight < 0 for weight in normalized_weights) or not any(
        normalized_weights
    ):
        raise DenseRetrievalError("fusion weights must be non-negative and not all zero")

    scores: dict[str, float] = {}
    best_hits: dict[str, SearchHit] = {}
    best_ranks: dict[str, int] = {}
    for ranking, weight in zip(rankings, normalized_weights, strict=True):
        seen: set[str] = set()
        for rank, hit in enumerate(ranking, start=1):
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (
                rrf_k + rank
            )
            if hit.chunk_id not in best_hits or rank < best_ranks[hit.chunk_id]:
                best_hits[hit.chunk_id] = hit
                best_ranks[hit.chunk_id] = rank

    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            best_ranks[chunk_id],
            chunk_id,
        ),
    )[:top_k]
    return tuple(
        replace(best_hits[chunk_id], rank=rank, score=scores[chunk_id])
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    )


class HybridRetriever:
    """Retrieve a broad candidate set and fuse dense plus lexical rankings."""

    def __init__(
        self,
        dense: Searcher,
        sparse: Searcher,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        if candidate_multiplier <= 0:
            raise DenseRetrievalError("candidate_multiplier must be positive")
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.weights = (dense_weight, sparse_weight)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        if top_k <= 0:
            raise DenseRetrievalError("top_k must be positive")
        candidate_k = top_k * self.candidate_multiplier
        dense_response = self.dense.search(
            query,
            top_k=candidate_k,
            filters=filters,
        )
        sparse_response = self.sparse.search(
            query,
            top_k=candidate_k,
            filters=filters,
        )
        if dense_response.collection_name != sparse_response.collection_name:
            raise DenseRetrievalError("dense and sparse collection names must match")
        hits = reciprocal_rank_fusion(
            [dense_response.hits, sparse_response.hits],
            weights=self.weights,
            rrf_k=self.rrf_k,
            top_k=top_k,
        )
        return SearchResponse(
            query=query,
            collection_name=dense_response.collection_name,
            embedding_model=(
                f"hybrid:{dense_response.embedding_model}+"
                f"{sparse_response.embedding_model}"
            ),
            hits=hits,
        )
