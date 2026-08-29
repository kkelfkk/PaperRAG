"""Data models returned by dense indexing and search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class IndexReport:
    """Summary of one document indexing operation."""

    collection_name: str
    document_id: str
    indexed_chunks: int
    vector_size: int
    embedding_model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked retrieval result with source citation metadata."""

    rank: int
    score: float
    chunk_id: str
    document_id: str
    source_file: str
    title: str
    page_number: int
    chunk_index: int
    section: str | None
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Serializable response for a dense retrieval query."""

    query: str
    collection_name: str
    embedding_model: str
    hits: tuple[SearchHit, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
