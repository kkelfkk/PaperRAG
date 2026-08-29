"""Tests for Qdrant-backed dense indexing and retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient

from app.chunking.models import DocumentChunk
from app.retrieval.dense import DenseRetrievalError, DenseRetriever


class KeywordEmbedder:
    """Small deterministic test embedding without model downloads."""

    model_name = "test/keyword-embedder"

    @staticmethod
    def _encode(text: str) -> list[float]:
        lowered = text.casefold()
        return [
            float(lowered.count("retrieval") + lowered.count("evidence")),
            float(lowered.count("agent") + lowered.count("tool")),
            0.1,
        ]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


class AlternateKeywordEmbedder(KeywordEmbedder):
    model_name = "test/alternate-keyword-embedder"


def _chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "doc-1",
    page_number: int = 1,
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_file=f"{document_id}.pdf",
        title=f"Paper {document_id}",
        page_number=page_number,
        chunk_index=chunk_index,
        section="Introduction",
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
    )


@pytest.fixture
def client() -> QdrantClient:
    qdrant = QdrantClient(location=":memory:")
    yield qdrant
    qdrant.close()


def test_index_and_search_returns_ranked_source_metadata(
    client: QdrantClient,
) -> None:
    retriever = DenseRetriever(client, KeywordEmbedder())
    chunks = [
        _chunk("retrieval", "Dense retrieval finds supporting evidence.", page_number=4),
        _chunk(
            "agent",
            "An agent selects a tool and executes an action.",
            page_number=7,
            chunk_index=1,
        ),
    ]

    report = retriever.index_document(chunks)
    response = retriever.search("How is evidence retrieved?", top_k=2)

    assert report.indexed_chunks == 2
    assert report.vector_size == 3
    assert report.embedding_model == "test/keyword-embedder"
    assert len(response.hits) == 2
    assert response.hits[0].chunk_id == "retrieval"
    assert response.hits[0].page_number == 4
    assert response.hits[0].source_file == "doc-1.pdf"
    assert response.hits[0].rank == 1
    assert response.hits[0].score >= response.hits[1].score


def test_reindex_replaces_old_document_chunks(client: QdrantClient) -> None:
    retriever = DenseRetriever(client, KeywordEmbedder())
    retriever.index_document(
        [
            _chunk("old-1", "retrieval evidence"),
            _chunk("old-2", "agent tool", chunk_index=1),
        ]
    )

    retriever.index_document([_chunk("new-1", "updated retrieval evidence")])
    response = retriever.search("retrieval", top_k=10)

    assert [hit.chunk_id for hit in response.hits] == ["new-1"]


def test_document_filter_limits_results(client: QdrantClient) -> None:
    retriever = DenseRetriever(client, KeywordEmbedder())
    retriever.index_document([_chunk("doc-1-chunk", "retrieval", document_id="doc-1")])
    retriever.index_document([_chunk("doc-2-chunk", "retrieval", document_id="doc-2")])

    response = retriever.search("retrieval", top_k=10, document_id="doc-2")

    assert [hit.document_id for hit in response.hits] == ["doc-2"]


def test_recreate_replaces_whole_collection(client: QdrantClient) -> None:
    retriever = DenseRetriever(client, KeywordEmbedder())
    retriever.index_document([_chunk("first", "retrieval", document_id="doc-1")])
    retriever.index_document(
        [_chunk("second", "agent", document_id="doc-2")], recreate=True
    )

    response = retriever.search("retrieval", top_k=10)

    assert [hit.chunk_id for hit in response.hits] == ["second"]


def test_collection_rejects_mixed_embedding_models(client: QdrantClient) -> None:
    DenseRetriever(client, KeywordEmbedder()).index_document(
        [_chunk("first", "retrieval")]
    )
    alternate = DenseRetriever(client, AlternateKeywordEmbedder())

    with pytest.raises(DenseRetrievalError, match="different embedding model"):
        alternate.search("retrieval")
    with pytest.raises(DenseRetrievalError, match="different embedding model"):
        alternate.index_document([_chunk("second", "retrieval")])

    report = alternate.index_document([_chunk("second", "retrieval")], recreate=True)
    assert report.embedding_model == "test/alternate-keyword-embedder"


def test_invalid_index_and_query_inputs_are_rejected(
    client: QdrantClient,
) -> None:
    retriever = DenseRetriever(client, KeywordEmbedder())

    with pytest.raises(DenseRetrievalError, match="empty chunk"):
        retriever.index_document([])
    with pytest.raises(DenseRetrievalError, match="does not exist"):
        retriever.search("query")

    retriever.index_document([_chunk("chunk", "retrieval")])
    with pytest.raises(DenseRetrievalError, match="query cannot be empty"):
        retriever.search("   ")
    with pytest.raises(DenseRetrievalError, match="top_k"):
        retriever.search("query", top_k=0)
