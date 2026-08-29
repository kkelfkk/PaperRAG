"""Data models emitted by the PDF ingestion pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DetectedHeading:
    """A likely section heading found on a physical PDF page."""

    text: str
    page_number: int
    level: int


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """Text and source metadata for one physical PDF page."""

    page_number: int
    text: str
    word_count: int
    headings: tuple[DetectedHeading, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Serializable representation of one parsed PDF document."""

    document_id: str
    source_file: str
    sha256: str
    title: str
    author: str | None
    page_count: int
    pages: tuple[ParsedPage, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""

        return asdict(self)
