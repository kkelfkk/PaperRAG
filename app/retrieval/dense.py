"""Qdrant-backed dense vector indexing and retrieval."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from app.chunking.models import DocumentChunk
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.models import IndexReport, SearchHit, SearchResponse

DEFAULT_COLLECTION = "paperrag_dense"


class DenseRetrievalError(ValueError):
    """Raised when indexing or retrieval configuration is invalid."""


def _embedding_text(chunk: DocumentChunk) -> str:
    fields = [f"Title: {chunk.title}"]
    if chunk.section:
        fields.append(f"Section: {chunk.section}")
    fields.append(chunk.text)
    return "\n".join(fields)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"paperrag:chunk:{chunk_id}"))


def _payload(chunk: DocumentChunk, embedding_model: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_file": chunk.source_file,
        "title": chunk.title,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "section": chunk.section,
        "text": chunk.text,
        "char_count": chunk.char_count,
        "word_count": chunk.word_count,
        "embedding_model": embedding_model,
    }


def _required_payload(payload: dict[str, Any] | None, key: str) -> Any:
    if payload is None or key not in payload:
        raise DenseRetrievalError(f"Qdrant result is missing payload field: {key}")
    return payload[key]


class DenseRetriever:
    """Index and query PaperRAG chunks using cosine similarity in Qdrant."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: EmbeddingProvider,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        if not collection_name.strip():
            raise DenseRetrievalError("collection_name cannot be empty")
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name

    def _stored_embedding_model(self) -> str | None:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1,
            with_payload=["embedding_model"],
            with_vectors=False,
        )
        if not points or not points[0].payload:
            return None
        value = points[0].payload.get("embedding_model")
        return str(value) if value is not None else None

    def _validate_embedding_model(self) -> None:
        stored_model = self._stored_embedding_model()
        if stored_model is not None and stored_model != self.embedder.model_name:
            raise DenseRetrievalError(
                "collection was indexed with a different embedding model: "
                f"collection={stored_model}, requested={self.embedder.model_name}. "
                "Choose the original model or recreate the collection."
            )

    def _ensure_collection(self, vector_size: int, *, recreate: bool) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if recreate and exists:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        info = self.client.get_collection(self.collection_name)
        configured_vectors = info.config.params.vectors
        configured_size = getattr(configured_vectors, "size", None)
        if configured_size != vector_size:
            raise DenseRetrievalError(
                "collection vector size does not match the embedding model: "
                f"collection={configured_size}, model={vector_size}. "
                "Use recreate=True or choose a different collection."
            )
        self._validate_embedding_model()

    def index_document(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        recreate: bool = False,
    ) -> IndexReport:
        """Replace one document's points and upsert its current chunks."""

        if not chunks:
            raise DenseRetrievalError("cannot index an empty chunk list")
        document_ids = {chunk.document_id for chunk in chunks}
        if len(document_ids) != 1:
            raise DenseRetrievalError("index_document accepts chunks from one document")

        vectors = self.embedder.embed_documents(
            [_embedding_text(chunk) for chunk in chunks]
        )
        if len(vectors) != len(chunks):
            raise DenseRetrievalError(
                "embedding count does not match the number of chunks"
            )
        if not vectors or not vectors[0]:
            raise DenseRetrievalError("embedding provider returned an empty vector")

        vector_size = len(vectors[0])
        if any(len(vector) != vector_size for vector in vectors):
            raise DenseRetrievalError("embedding vectors have inconsistent dimensions")
        self._ensure_collection(vector_size, recreate=recreate)

        document_id = next(iter(document_ids))
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=vector,
                    payload=_payload(chunk, self.embedder.model_name),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
            wait=True,
        )
        return IndexReport(
            collection_name=self.collection_name,
            document_id=document_id,
            indexed_chunks=len(chunks),
            vector_size=vector_size,
            embedding_model=self.embedder.model_name,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> SearchResponse:
        """Return the top dense results, optionally filtered to one document."""

        if not query.strip():
            raise DenseRetrievalError("query cannot be empty")
        if top_k <= 0:
            raise DenseRetrievalError("top_k must be positive")
        if not self.client.collection_exists(self.collection_name):
            raise DenseRetrievalError(
                f"Qdrant collection does not exist: {self.collection_name}"
            )
        self._validate_embedding_model()

        query_filter = None
        if document_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embedder.embed_query(query),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        hits = tuple(
            SearchHit(
                rank=rank,
                score=float(point.score),
                chunk_id=str(_required_payload(point.payload, "chunk_id")),
                document_id=str(_required_payload(point.payload, "document_id")),
                source_file=str(_required_payload(point.payload, "source_file")),
                title=str(_required_payload(point.payload, "title")),
                page_number=int(_required_payload(point.payload, "page_number")),
                chunk_index=int(_required_payload(point.payload, "chunk_index")),
                section=(
                    str(point.payload["section"])
                    if point.payload and point.payload.get("section") is not None
                    else None
                ),
                text=str(_required_payload(point.payload, "text")),
            )
            for rank, point in enumerate(response.points, start=1)
        )
        return SearchResponse(
            query=query,
            collection_name=self.collection_name,
            embedding_model=self.embedder.model_name,
            hits=hits,
        )
