"""Data models for evidence-grounded answers and citations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Limits for assembling retrieved context."""

    max_context_chars: int = 12000

    def __post_init__(self) -> None:
        if self.max_context_chars < 500:
            raise ValueError("max_context_chars must be at least 500")


@dataclass(frozen=True, slots=True)
class Citation:
    """A source that was explicitly cited in the generated answer."""

    source_id: str
    chunk_id: str
    document_id: str
    source_file: str
    title: str
    page_number: int
    section: str | None


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """Validated answer plus its verifiable citation metadata."""

    query: str
    answer: str
    abstained: bool
    model: str
    retrieved_count: int
    citations: tuple[Citation, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
