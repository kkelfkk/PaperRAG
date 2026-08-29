"""Data models produced by the document chunking pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Character-based limits for deterministic recursive chunking."""

    max_chunk_chars: int = 1200
    overlap_chars: int = 200

    def __post_init__(self) -> None:
        if self.max_chunk_chars < 64:
            raise ValueError("max_chunk_chars must be at least 64")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative")
        if self.overlap_chars > self.max_chunk_chars // 2:
            raise ValueError("overlap_chars cannot exceed half of max_chunk_chars")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A retrieval-ready text unit with verifiable source metadata."""

    chunk_id: str
    document_id: str
    source_file: str
    title: str
    page_number: int
    chunk_index: int
    section: str | None
    text: str
    char_count: int
    word_count: int


@dataclass(frozen=True, slots=True)
class ChunkedDocument:
    """All retrieval chunks emitted for one parsed document."""

    document_id: str
    source_file: str
    title: str
    chunk_count: int
    config: ChunkingConfig
    chunks: tuple[DocumentChunk, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""

        return asdict(self)
