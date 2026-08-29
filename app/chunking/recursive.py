"""Page-aware, section-aware recursive text chunking."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from app.chunking.models import ChunkedDocument, ChunkingConfig, DocumentChunk
from app.ingestion.models import DetectedHeading, ParsedDocument, ParsedPage

_SEPARATORS = ("\n\n", "\n", " ")


def _recursive_segments(
    text: str,
    limit: int,
    separators: tuple[str, ...] = _SEPARATORS,
) -> list[str]:
    """Split text by paragraph, line, word, then character boundaries."""

    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= limit:
        return [normalized]
    if not separators:
        return [
            normalized[start : start + limit].strip()
            for start in range(0, len(normalized), limit)
            if normalized[start : start + limit].strip()
        ]

    separator, remaining = separators[0], separators[1:]
    if separator not in normalized:
        return _recursive_segments(normalized, limit, remaining)

    parts = [part.strip() for part in normalized.split(separator) if part.strip()]
    segments: list[str] = []
    current = ""

    for part in parts:
        candidate = separator.join((current, part)) if current else part
        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            segments.append(current)
            current = ""

        if len(part) > limit:
            segments.extend(_recursive_segments(part, limit, remaining))
        else:
            current = part

    if current:
        segments.append(current)
    return segments


def _overlap_tail(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text

    tail = text[-limit:]
    for separator in ("\n", " "):
        boundary = tail.find(separator)
        if 0 <= boundary < len(tail) // 2:
            candidate = tail[boundary + len(separator) :].strip()
            if candidate:
                return candidate
    return tail.strip()


def split_text(text: str, config: ChunkingConfig) -> tuple[str, ...]:
    """Split one section and add bounded overlap between adjacent chunks."""

    separator_cost = 1 if config.overlap_chars else 0
    content_limit = config.max_chunk_chars - config.overlap_chars - separator_cost
    base_segments = _recursive_segments(text, content_limit)
    if not base_segments:
        return ()

    chunks: list[str] = []
    previous = ""
    for segment in base_segments:
        overlap = _overlap_tail(previous, config.overlap_chars)
        combined = f"{overlap}\n{segment}" if overlap else segment
        chunks.append(combined)
        previous = segment
    return tuple(chunks)


def _heading_lookup(headings: Iterable[DetectedHeading]) -> dict[str, str]:
    return {" ".join(item.text.split()).casefold(): item.text for item in headings}


def _page_sections(
    page: ParsedPage,
    inherited_section: str | None,
) -> tuple[list[tuple[str | None, str]], str | None]:
    """Group a page's lines under detected headings without crossing pages."""

    headings = _heading_lookup(page.headings)
    active_section = inherited_section
    blocks: list[tuple[str | None, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            blocks.append((active_section, text))
        buffer.clear()

    for raw_line in page.text.splitlines():
        normalized = " ".join(raw_line.split())
        detected = headings.get(normalized.casefold())
        if detected is not None:
            flush()
            active_section = detected
            continue
        buffer.append(raw_line)
    flush()
    return blocks, active_section


def _chunk_id(
    document_id: str,
    page_number: int,
    chunk_index: int,
    section: str | None,
    text: str,
) -> str:
    identity = "\x1f".join(
        (document_id, str(page_number), str(chunk_index), section or "", text)
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def chunk_document(
    document: ParsedDocument,
    config: ChunkingConfig | None = None,
) -> ChunkedDocument:
    """Create retrieval chunks while preserving page and section provenance."""

    selected_config = config or ChunkingConfig()
    chunks: list[DocumentChunk] = []
    warnings = list(document.warnings)
    active_section: str | None = None

    for page in document.pages:
        blocks, active_section = _page_sections(page, active_section)
        for section, block_text in blocks:
            for text in split_text(block_text, selected_config):
                chunk_index = len(chunks)
                chunks.append(
                    DocumentChunk(
                        chunk_id=_chunk_id(
                            document.document_id,
                            page.page_number,
                            chunk_index,
                            section,
                            text,
                        ),
                        document_id=document.document_id,
                        source_file=document.source_file,
                        title=document.title,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        section=section,
                        text=text,
                        char_count=len(text),
                        word_count=len(text.split()),
                    )
                )

    if not chunks:
        warnings.append("Document produced no chunks because it has no extractable text.")

    return ChunkedDocument(
        document_id=document.document_id,
        source_file=document.source_file,
        title=document.title,
        chunk_count=len(chunks),
        config=selected_config,
        chunks=tuple(chunks),
        warnings=tuple(dict.fromkeys(warnings)),
    )
