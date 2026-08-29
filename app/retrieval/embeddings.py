"""Replaceable embedding providers for retrieval."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal interface required by the dense retrieval pipeline."""

    @property
    def model_name(self) -> str:
        """Return the stable model identifier used for these vectors."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for indexing."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query."""


class FastEmbedProvider:
    """CPU-friendly ONNX embeddings backed by Qdrant FastEmbed."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        from fastembed import TextEmbedding

        self._model_name = model_name
        # This project starts with FastEmbed's current mean-pooling behavior, so
        # its compatibility warning about older CLS-based vectors is not relevant.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The model .* now uses mean pooling instead of CLS embedding.*",
                category=UserWarning,
            )
            self._model = TextEmbedding(model_name=model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("query cannot be empty")
        vectors = list(self._model.query_embed([text]))
        if len(vectors) != 1:
            raise RuntimeError("embedding provider did not return exactly one query vector")
        return vectors[0].tolist()
