"""Second-stage retrieval wrapper that reranks a broad candidate set."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from app.reranking.cross_encoder import CrossEncoderProvider, RerankingError
from app.retrieval.models import SearchHit, SearchResponse


class Searcher(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> SearchResponse: ...


def _passage(hit: SearchHit) -> str:
    fields = [f"Title: {hit.title}"]
    if hit.section:
        fields.append(f"Section: {hit.section}")
    fields.append(hit.text)
    return "\n".join(fields)


class RerankingRetriever:
    """Retrieve candidates, score query-passage pairs, and return the new Top-K."""

    def __init__(
        self,
        candidate_retriever: Searcher,
        reranker: CrossEncoderProvider,
        *,
        candidate_multiplier: int = 4,
        minimum_candidates: int = 20,
    ) -> None:
        if candidate_multiplier <= 0:
            raise RerankingError("candidate_multiplier must be positive")
        if minimum_candidates <= 0:
            raise RerankingError("minimum_candidates must be positive")
        self.candidate_retriever = candidate_retriever
        self.reranker = reranker
        self.candidate_multiplier = candidate_multiplier
        self.minimum_candidates = minimum_candidates

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> SearchResponse:
        if top_k <= 0:
            raise RerankingError("top_k must be positive")
        candidate_k = max(top_k * self.candidate_multiplier, self.minimum_candidates)
        response = self.candidate_retriever.search(
            query,
            top_k=candidate_k,
            document_id=document_id,
        )
        if not response.hits:
            return SearchResponse(
                query=query,
                collection_name=response.collection_name,
                embedding_model=(
                    f"{response.embedding_model}|reranker:{self.reranker.model_name}"
                ),
                hits=(),
            )

        scores = self.reranker.score(query, [_passage(hit) for hit in response.hits])
        if len(scores) != len(response.hits):
            raise RerankingError("reranker score count does not match candidate count")
        ranked = sorted(
            zip(response.hits, scores, strict=True),
            key=lambda item: (-item[1], item[0].rank, item[0].chunk_id),
        )[:top_k]
        hits = tuple(
            replace(hit, rank=rank, score=score)
            for rank, (hit, score) in enumerate(ranked, start=1)
        )
        return SearchResponse(
            query=query,
            collection_name=response.collection_name,
            embedding_model=(
                f"{response.embedding_model}|reranker:{self.reranker.model_name}"
            ),
            hits=hits,
        )
