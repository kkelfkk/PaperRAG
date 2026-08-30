"""Tests for deterministic multi-paper query decomposition."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.retrieval.decomposition import (
    AliasQueryDecomposer,
    DecomposedRetriever,
    DocumentTarget,
    load_query_decomposer,
)
from app.retrieval.filters import SearchFilters
from app.retrieval.models import SearchHit, SearchResponse

TARGETS = (
    DocumentTarget("rag-doc", "Retrieval-Augmented Generation", ("RAG",)),
    DocumentTarget("react-doc", "ReAct", ("ReAct",)),
    DocumentTarget("selfrag-doc", "Self-RAG", ("Self-RAG",)),
    DocumentTarget("crag-doc", "Corrective RAG", ("CRAG",)),
)


def _hit(document_id: str, rank: int) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=1.0 / rank,
        chunk_id=f"{document_id}-{rank}",
        document_id=document_id,
        source_file=f"{document_id}.pdf",
        title=document_id,
        page_number=rank,
        chunk_index=rank - 1,
        section=None,
        text=f"passage {rank}",
    )


class FakeSearcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, SearchFilters | None]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        self.calls.append((query, top_k, filters))
        document_id = filters.document_id if filters and filters.document_id else "all"
        hits = tuple(_hit(document_id, rank) for rank in range(1, top_k + 1))
        return SearchResponse(query, "papers", "test-model", hits)


def test_longer_aliases_do_not_also_match_rag() -> None:
    decomposer = AliasQueryDecomposer(TARGETS)

    plans = decomposer.decompose("Self-RAG 和 CRAG 如何评价证据？")

    assert [plan.target.document_id for plan in plans if plan.target] == [
        "selfrag-doc",
        "crag-doc",
    ]


def test_collective_marker_expands_to_every_paper() -> None:
    decomposer = AliasQueryDecomposer(TARGETS)

    plans = decomposer.decompose("四篇论文分别解决了什么问题？")

    assert [plan.target.document_id for plan in plans if plan.target] == [
        target.document_id for target in TARGETS
    ]


def test_single_paper_or_explicit_filter_is_not_decomposed() -> None:
    decomposer = AliasQueryDecomposer(TARGETS)

    assert len(decomposer.decompose("RAG 如何检索？")) == 1
    filters = SearchFilters(document_id="rag-doc", section="Methods")
    assert decomposer.decompose("比较 RAG 与 CRAG", filters=filters)[0].filters == filters


def test_decomposed_retriever_balances_documents_and_preserves_filters() -> None:
    searcher = FakeSearcher()
    retriever = DecomposedRetriever(searcher, AliasQueryDecomposer(TARGETS))
    filters = SearchFilters(section="Methods", page_from=2)

    response = retriever.search("比较 RAG、Self-RAG 和 CRAG", top_k=6, filters=filters)

    assert len(searcher.calls) == 3
    assert {call[2].document_id for call in searcher.calls if call[2]} == {
        "rag-doc",
        "selfrag-doc",
        "crag-doc",
    }
    assert all(call[2].section == "Methods" for call in searcher.calls if call[2])
    assert all(call[2].page_from == 2 for call in searcher.calls if call[2])
    assert {hit.document_id for hit in response.hits[:3]} == {
        "rag-doc",
        "selfrag-doc",
        "crag-doc",
    }
    assert response.embedding_model == "test-model|query_decomposition"
    assert response.query == "比较 RAG、Self-RAG 和 CRAG"


def test_decomposed_retriever_rejects_changed_configuration() -> None:
    class ChangingSearcher(FakeSearcher):
        def search(
            self,
            query: str,
            *,
            top_k: int = 5,
            filters: SearchFilters | None = None,
        ) -> SearchResponse:
            response = super().search(query, top_k=top_k, filters=filters)
            if filters and filters.document_id == "crag-doc":
                return replace(response, collection_name="other")
            return response

    retriever = DecomposedRetriever(ChangingSearcher(), AliasQueryDecomposer(TARGETS))

    with pytest.raises(ValueError, match="collection names"):
        retriever.search("比较 RAG 和 CRAG")


def test_passthrough_keeps_a_stable_strategy_identifier() -> None:
    retriever = DecomposedRetriever(FakeSearcher(), AliasQueryDecomposer(TARGETS))

    response = retriever.search("RAG 如何检索？")

    assert response.embedding_model == "test-model|query_decomposition"


def test_manifest_loader_supports_explicit_ids_and_aliases(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps(
            {
                "papers": [
                    {
                        "document_id": "paper-a",
                        "title": "Paper Alpha",
                        "aliases": ["Alpha", "论文甲"],
                    },
                    {
                        "document_id": "paper-b",
                        "title": "Paper Beta",
                        "aliases": ["Beta", "论文乙"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    plans = load_query_decomposer(manifest).decompose("比较论文甲与论文乙")

    assert [plan.target.document_id for plan in plans if plan.target] == [
        "paper-a",
        "paper-b",
    ]
