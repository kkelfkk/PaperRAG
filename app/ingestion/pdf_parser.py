"""Extract page-addressable text and metadata from text-based PDF papers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pdfplumber

from app.ingestion.models import DetectedHeading, ParsedDocument, ParsedPage


class PDFParseError(ValueError):
    """Raised when a PDF cannot be validated or parsed."""


_NUMBERED_HEADING = re.compile(
    r"^(?P<number>\d+(?:\.\d+){0,4})\.?\s+(?P<title>\S.{0,119})$"
)
_COMMON_HEADINGS = {
    "abstract",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "conclusion",
    "conclusions",
    "discussion",
    "evaluation",
    "experiments",
    "introduction",
    "limitations",
    "method",
    "methodology",
    "references",
    "related work",
    "results",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""

    normalized = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.splitlines()]

    output: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        output.append(line)
        previous_blank = is_blank
    return "\n".join(output).strip()


def _heading_level(number: str | None) -> int:
    return number.count(".") + 1 if number else 1


def detect_headings(text: str, page_number: int) -> tuple[DetectedHeading, ...]:
    """Detect conservative academic heading candidates from extracted lines."""

    headings: list[DetectedHeading] = []
    seen: set[str] = set()

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line or len(line) > 140 or len(line.split()) > 18:
            continue

        numbered = _NUMBERED_HEADING.match(line)
        common = line.casefold().rstrip(":") in _COMMON_HEADINGS
        upper = (
            line.isupper()
            and 2 <= len(line.split()) <= 10
            and any(character.isalpha() for character in line)
        )
        if not (numbered or common or upper):
            continue

        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        number = numbered.group("number") if numbered else None
        headings.append(
            DetectedHeading(
                text=line,
                page_number=page_number,
                level=_heading_level(number),
            )
        )

    return tuple(headings)


def _clean_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        normalized = " ".join(str(value).split()).strip()
        if normalized:
            cleaned[str(key).casefold()] = normalized
    return cleaned


def _validate_pdf(path: Path) -> None:
    if not path.exists():
        raise PDFParseError(f"PDF file does not exist: {path}")
    if not path.is_file():
        raise PDFParseError(f"PDF path is not a file: {path}")
    if path.suffix.casefold() != ".pdf":
        raise PDFParseError(f"Expected a .pdf file: {path}")

    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise PDFParseError(f"File does not have a valid PDF header: {path}")


def parse_pdf(pdf_path: str | Path) -> ParsedDocument:
    """Parse a text-based PDF while preserving physical page numbering."""

    path = Path(pdf_path).expanduser().resolve()
    _validate_pdf(path)
    fingerprint = _sha256(path)
    warnings: list[str] = []
    pages: list[ParsedPage] = []

    try:
        with pdfplumber.open(path) as pdf:
            metadata = _clean_metadata(pdf.metadata)
            for page_number, page in enumerate(pdf.pages, start=1):
                text = _normalize_text(page.extract_text())
                if not text:
                    warnings.append(
                        f"Page {page_number} contains no extractable text; OCR may be required."
                    )
                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        text=text,
                        word_count=len(text.split()),
                        headings=detect_headings(text, page_number),
                    )
                )
    except PDFParseError:
        raise
    except Exception as exc:
        raise PDFParseError(f"Could not parse PDF {path.name}: {exc}") from exc

    if not pages:
        raise PDFParseError(f"PDF contains no pages: {path}")

    title = metadata.get("title") or path.stem
    author = metadata.get("author")
    return ParsedDocument(
        document_id=fingerprint[:16],
        source_file=path.name,
        sha256=fingerprint,
        title=title,
        author=author,
        page_count=len(pages),
        pages=tuple(pages),
        warnings=tuple(warnings),
    )
