"""Deterministic query decomposition for multi-paper comparison questions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from app.retrieval.dense import DenseRetrievalError
from app.retrieval.filters import SearchFilters
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.models import SearchResponse


class Searcher(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResponse: ...


@dataclass(frozen=True, slots=True)
class DocumentTarget:
    """One indexed paper and the names users may use to refer to it."""

    document_id: str
    title: str
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document target ID cannot be empty")
        if not self.title.strip():
            raise ValueError("document target title cannot be empty")
        if not self.aliases or any(not alias.strip() for alias in self.aliases):
            raise ValueError("document target aliases cannot be empty")


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """A focused subquery and its document-aware retrieval scope."""

    query: str
    filters: SearchFilters | None
    target: DocumentTarget | None = None


class AliasQueryDecomposer:
    """Split questions that explicitly name two or more indexed papers."""

    def __init__(
        self,
        targets: tuple[DocumentTarget, ...],
        *,
        collective_markers: tuple[str, ...] = (
            "四篇论文",
            "四种方法",
            "all papers",
            "all four papers",
        ),
    ) -> None:
        if len(targets) < 2:
            raise ValueError("query decomposition requires at least two document targets")
        document_ids = [target.document_id for target in targets]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document target IDs must be unique")
        self.targets = targets
        self.collective_markers = collective_markers

    @staticmethod
    def _overlaps(start: int, end: int, occupied: list[tuple[int, int]]) -> bool:
        return any(start < other_end and end > other_start for other_start, other_end in occupied)

    def _mentioned_targets(self, query: str) -> tuple[DocumentTarget, ...]:
        folded = query.casefold()
        if any(marker.casefold() in folded for marker in self.collective_markers):
            return self.targets

        occurrences: list[tuple[int, int, DocumentTarget]] = []
        for target in self.targets:
            for alias in target.aliases:
                alias_folded = alias.casefold()
                start = folded.find(alias_folded)
                while start >= 0:
                    occurrences.append((start, start + len(alias_folded), target))
                    start = folded.find(alias_folded, start + 1)

        # Prefer longer aliases so "Self-RAG" and "CRAG" do not also match "RAG".
        selected_ids: set[str] = set()
        occupied: list[tuple[int, int]] = []
        for start, end, target in sorted(
            occurrences,
            key=lambda item: (-(item[1] - item[0]), item[0], item[2].document_id),
        ):
            if target.document_id in selected_ids or self._overlaps(start, end, occupied):
                continue
            selected_ids.add(target.document_id)
            occupied.append((start, end))
        return tuple(
            target for target in self.targets if target.document_id in selected_ids
        )

    @staticmethod
    def _target_filters(
        filters: SearchFilters | None,
        document_id: str,
    ) -> SearchFilters:
        return SearchFilters(
            document_id=document_id,
            section=filters.section if filters else None,
            page_from=filters.page_from if filters else None,
            page_to=filters.page_to if filters else None,
        )

    def decompose(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
    ) -> tuple[QueryPlan, ...]:
        """Return one plan per named paper, or the unchanged query when not applicable."""

        if not query.strip():
            raise ValueError("query cannot be empty")
        if filters and filters.document_id:
            return (QueryPlan(query=query, filters=filters),)

        targets = self._mentioned_targets(query)
        if len(targets) < 2:
            return (QueryPlan(query=query, filters=filters),)
        return tuple(
            QueryPlan(
                query=f"{query}\nFocus on paper: {target.title}.",
                filters=self._target_filters(filters, target.document_id),
                target=target,
            )
            for target in targets
        )


class DecomposedRetriever:
    """Retrieve each paper-focused subquery and fuse the balanced rankings."""

    def __init__(
        self,
        searcher: Searcher,
        decomposer: AliasQueryDecomposer,
        *,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")
        self.searcher = searcher
        self.decomposer = decomposer
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        if top_k <= 0:
            raise DenseRetrievalError("top_k must be positive")
        plans = self.decomposer.decompose(query, filters=filters)
        if len(plans) == 1:
            response = self.searcher.search(query, top_k=top_k, filters=filters)
            return replace(
                response,
                query=query,
                embedding_model=f"{response.embedding_model}|query_decomposition",
            )

        responses = tuple(
            self.searcher.search(plan.query, top_k=top_k, filters=plan.filters)
            for plan in plans
        )
        collection_name = responses[0].collection_name
        embedding_model = responses[0].embedding_model
        if any(response.collection_name != collection_name for response in responses):
            raise DenseRetrievalError("subquery collection names must match")
        if any(response.embedding_model != embedding_model for response in responses):
            raise DenseRetrievalError("subquery retrieval configurations must match")

        return SearchResponse(
            query=query,
            collection_name=collection_name,
            embedding_model=f"{embedding_model}|query_decomposition",
            hits=reciprocal_rank_fusion(
                [response.hits for response in responses],
                rrf_k=self.rrf_k,
                top_k=top_k,
            ),
        )
