"""Tests for lexical BM25 retrieval over Qdrant payloads."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient

from app.chunking.models import DocumentChunk
from app.retrieval.bm25 import BM25_MODEL_NAME, BM25Retriever, tokenize
from app.retrieval.dense import DenseRetriever


class ConstantEmbedder:
    model_name = "test/constant"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "doc-1",
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_file=f"{document_id}.pdf",
        title="Retrieval Paper",
        page_number=chunk_index + 1,
        chunk_index=chunk_index,
        section="Methods",
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
    )


@pytest.fixture
def client() -> QdrantClient:
    qdrant = QdrantClient(location=":memory:")
    yield qdrant
    qdrant.close()


def test_tokenize_supports_technical_terms_and_chinese_bigrams() -> None:
    assert tokenize("BM25 cross-encoder RRF") == ("bm25", "cross-encoder", "rrf")
    assert tokenize("混合检索") == ("混合", "合检", "检索")


def test_bm25_ranks_exact_technical_term_first(client: QdrantClient) -> None:
    dense = DenseRetriever(client, ConstantEmbedder())
    dense.index_document(
        [
            _chunk("generic", "Semantic retrieval finds related passages."),
            _chunk(
                "exact",
                "Reciprocal Rank Fusion combines BM25 and dense rankings.",
                chunk_index=1,
            ),
        ]
    )

    response = BM25Retriever(client, dense.collection_name).search("BM25", top_k=2)

    assert response.embedding_model == BM25_MODEL_NAME
    assert [hit.chunk_id for hit in response.hits] == ["exact"]
    assert response.hits[0].score > 0
    assert response.hits[0].rank == 1


def test_bm25_document_filter_and_no_match(client: QdrantClient) -> None:
    dense = DenseRetriever(client, ConstantEmbedder())
    dense.index_document([_chunk("first", "rarekeyword", document_id="doc-1")])
    dense.index_document([_chunk("second", "rarekeyword", document_id="doc-2")])
    retriever = BM25Retriever(client, dense.collection_name)

    filtered = retriever.search("rarekeyword", document_id="doc-2")
    missing = retriever.search("not-present-anywhere")

    assert [hit.chunk_id for hit in filtered.hits] == ["second"]
    assert missing.hits == ()
