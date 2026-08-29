"""Cross-encoder scoring providers with lazy model loading."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class RerankingError(ValueError):
    """Raised when a reranker cannot produce valid relevance scores."""


class CrossEncoderProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class SentenceTransformersCrossEncoder:
    """Lazy sentence-transformers adapter for local cross-encoder inference."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise RerankingError("reranker model_name cannot be empty")
        if max_length <= 0:
            raise RerankingError("reranker max_length must be positive")
        self._model_name = model_name
        self.max_length = max_length
        self.device = device
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankingError(
                "reranking dependencies are not installed; run "
                "`uv sync --extra dev --extra rerank`"
            ) from exc
        self._model = CrossEncoder(
            self._model_name,
            max_length=self.max_length,
            device=self.device,
            trust_remote_code=False,
        )
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not query.strip():
            raise RerankingError("query cannot be empty")
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        predictions = self._load_model().predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = [float(score) for score in predictions]
        if len(scores) != len(passages):
            raise RerankingError("reranker score count does not match candidate count")
        if not all(math.isfinite(score) for score in scores):
            raise RerankingError("reranker returned a non-finite score")
        return scores
