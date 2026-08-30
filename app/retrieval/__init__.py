"""Information retrieval package."""

from app.retrieval.bm25 import BM25_MODEL_NAME, BM25Retriever, tokenize
from app.retrieval.decomposition import (
    AliasQueryDecomposer,
    DecomposedRetriever,
    DocumentTarget,
    QueryPlan,
    load_query_decomposer,
)
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
from app.retrieval.filters import SearchFilters
from app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.models import IndexReport, SearchHit, SearchResponse

__all__ = [
    "BM25_MODEL_NAME",
    "DEFAULT_COLLECTION",
    "DEFAULT_EMBEDDING_MODEL",
    "AliasQueryDecomposer",
    "BM25Retriever",
    "DecomposedRetriever",
    "DenseRetrievalError",
    "DenseRetriever",
    "DocumentTarget",
    "EmbeddingProvider",
    "FastEmbedProvider",
    "HybridRetriever",
    "IndexReport",
    "QueryPlan",
    "SearchFilters",
    "SearchHit",
    "SearchResponse",
    "load_query_decomposer",
    "reciprocal_rank_fusion",
    "tokenize",
]
