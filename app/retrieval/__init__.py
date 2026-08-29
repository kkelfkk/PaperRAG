"""Information retrieval package."""

from app.retrieval.dense import (
    DEFAULT_COLLECTION,
    DenseRetrievalError,
    DenseRetriever,
)
from app.retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingProvider,
    FastEmbedProvider,
)
from app.retrieval.models import IndexReport, SearchHit, SearchResponse

__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_EMBEDDING_MODEL",
    "DenseRetrievalError",
    "DenseRetriever",
    "EmbeddingProvider",
    "FastEmbedProvider",
    "IndexReport",
    "SearchHit",
    "SearchResponse",
]
