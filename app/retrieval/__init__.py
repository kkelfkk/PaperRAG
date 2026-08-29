"""Information retrieval package."""

from app.retrieval.bm25 import BM25_MODEL_NAME, BM25Retriever, tokenize
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
from app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.models import IndexReport, SearchHit, SearchResponse

__all__ = [
    "BM25_MODEL_NAME",
    "DEFAULT_COLLECTION",
    "DEFAULT_EMBEDDING_MODEL",
    "BM25Retriever",
    "DenseRetrievalError",
    "DenseRetriever",
    "EmbeddingProvider",
    "FastEmbedProvider",
    "HybridRetriever",
    "IndexReport",
    "SearchHit",
    "SearchResponse",
    "reciprocal_rank_fusion",
    "tokenize",
]
