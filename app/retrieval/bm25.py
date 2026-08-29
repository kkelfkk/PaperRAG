"""Dependency-free BM25 retrieval over chunk payloads stored in Qdrant."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient

from app.retrieval.dense import DenseRetrievalError
from app.retrieval.filters import SearchFilters
from app.retrieval.models import SearchHit, SearchResponse

BM25_MODEL_NAME = "bm25-v1"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u3400-\u9fff]+")


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize Latin terms and Chinese character bigrams deterministically."""

    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text.casefold()):
        value = match.group()
        if "\u3400" <= value[0] <= "\u9fff":
            if len(value) == 1:
                tokens.append(value)
            else:
                tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
        else:
            tokens.append(value)
    return tuple(tokens)


def _required(payload: dict[str, Any] | None, key: str) -> Any:
    if payload is None or key not in payload:
        raise DenseRetrievalError(f"Qdrant result is missing payload field: {key}")
    return payload[key]


@dataclass(frozen=True, slots=True)
class _BM25Document:
    payload: dict[str, Any]
    term_frequencies: Counter[str]
    length: int


class BM25Retriever:
    """Rank Qdrant chunk payloads with Okapi BM25."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        scroll_batch_size: int = 256,
    ) -> None:
        if not collection_name.strip():
            raise DenseRetrievalError("collection_name cannot be empty")
        if k1 <= 0 or not 0 <= b <= 1:
            raise DenseRetrievalError("BM25 requires k1 > 0 and 0 <= b <= 1")
        if scroll_batch_size <= 0:
            raise DenseRetrievalError("scroll_batch_size must be positive")
        self.client = client
        self.collection_name = collection_name
        self.k1 = k1
        self.b = b
        self.scroll_batch_size = scroll_batch_size

    def _load_documents(self, filters: SearchFilters | None) -> list[_BM25Document]:
        query_filter = filters.to_qdrant() if filters is not None else None

        documents: list[_BM25Document] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=self.scroll_batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                title = str(_required(payload, "title"))
                section = payload.get("section")
                text = str(_required(payload, "text"))
                searchable = "\n".join(
                    [title, title, str(section) if section is not None else "", text]
                )
                terms = tokenize(searchable)
                documents.append(
                    _BM25Document(
                        payload=payload,
                        term_frequencies=Counter(terms),
                        length=len(terms),
                    )
                )
            if offset is None:
                break
        return documents

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        if not query.strip():
            raise DenseRetrievalError("query cannot be empty")
        if top_k <= 0:
            raise DenseRetrievalError("top_k must be positive")
        if not self.client.collection_exists(self.collection_name):
            raise DenseRetrievalError(
                f"Qdrant collection does not exist: {self.collection_name}"
            )
        query_terms = tuple(dict.fromkeys(tokenize(query)))
        if not query_terms:
            raise DenseRetrievalError("query does not contain searchable terms")

        documents = self._load_documents(filters)
        if not documents:
            return SearchResponse(
                query=query,
                collection_name=self.collection_name,
                embedding_model=BM25_MODEL_NAME,
                hits=(),
            )
        average_length = (
            sum(document.length for document in documents) / len(documents) or 1.0
        )
        document_frequencies = {
            term: sum(term in document.term_frequencies for document in documents)
            for term in query_terms
        }
        scored: list[tuple[float, _BM25Document]] = []
        for document in documents:
            score = 0.0
            for term in query_terms:
                frequency = document.term_frequencies.get(term, 0)
                if not frequency:
                    continue
                frequency_docs = document_frequencies[term]
                inverse_frequency = math.log(
                    1 + (len(documents) - frequency_docs + 0.5) / (frequency_docs + 0.5)
                )
                normalization = frequency + self.k1 * (
                    1 - self.b + self.b * document.length / average_length
                )
                score += inverse_frequency * frequency * (self.k1 + 1) / normalization
            if score > 0:
                scored.append((score, document))

        scored.sort(
            key=lambda item: (-item[0], str(_required(item[1].payload, "chunk_id")))
        )
        hits = tuple(
            SearchHit(
                rank=rank,
                score=score,
                chunk_id=str(_required(document.payload, "chunk_id")),
                document_id=str(_required(document.payload, "document_id")),
                source_file=str(_required(document.payload, "source_file")),
                title=str(_required(document.payload, "title")),
                page_number=int(_required(document.payload, "page_number")),
                chunk_index=int(_required(document.payload, "chunk_index")),
                section=(
                    str(document.payload["section"])
                    if document.payload.get("section") is not None
                    else None
                ),
                text=str(_required(document.payload, "text")),
            )
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        )
        return SearchResponse(
            query=query,
            collection_name=self.collection_name,
            embedding_model=BM25_MODEL_NAME,
            hits=hits,
        )
