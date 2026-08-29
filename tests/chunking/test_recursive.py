"""Tests for page-aware recursive document chunking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from reportlab.pdfgen.canvas import Canvas

from app.chunking.cli import main
from app.chunking.models import ChunkingConfig
from app.chunking.recursive import chunk_document, split_text
from app.ingestion.models import DetectedHeading, ParsedDocument, ParsedPage


def _document() -> ParsedDocument:
    introduction = " ".join(f"introduction-{index}" for index in range(80))
    continuation = " ".join(f"continuation-{index}" for index in range(30))
    method = " ".join(f"method-{index}" for index in range(45))
    return ParsedDocument(
        document_id="document-123",
        source_file="paper.pdf",
        sha256="a" * 64,
        title="Test Paper",
        author="Researcher",
        page_count=3,
        pages=(
            ParsedPage(
                page_number=1,
                text=f"1 Introduction\n{introduction}",
                word_count=81,
                headings=(DetectedHeading("1 Introduction", 1, 1),),
            ),
            ParsedPage(
                page_number=2,
                text=continuation,
                word_count=30,
            ),
            ParsedPage(
                page_number=3,
                text=f"2 Method\n{method}",
                word_count=46,
                headings=(DetectedHeading("2 Method", 3, 1),),
            ),
        ),
    )


def test_chunk_document_preserves_page_and_inherited_section() -> None:
    result = chunk_document(
        _document(), ChunkingConfig(max_chunk_chars=220, overlap_chars=40)
    )

    assert result.chunk_count == len(result.chunks)
    assert result.chunk_count > 3
    assert {chunk.page_number for chunk in result.chunks} == {1, 2, 3}
    assert all(chunk.section == "1 Introduction" for chunk in result.chunks if chunk.page_number <= 2)
    assert all(chunk.section == "2 Method" for chunk in result.chunks if chunk.page_number == 3)
    assert all(chunk.source_file == "paper.pdf" for chunk in result.chunks)
    assert all(chunk.title == "Test Paper" for chunk in result.chunks)


def test_chunks_respect_max_size_and_have_stable_ids() -> None:
    config = ChunkingConfig(max_chunk_chars=180, overlap_chars=30)

    first = chunk_document(_document(), config)
    second = chunk_document(_document(), config)

    assert all(chunk.char_count <= config.max_chunk_chars for chunk in first.chunks)
    assert [chunk.chunk_id for chunk in first.chunks] == [
        chunk.chunk_id for chunk in second.chunks
    ]
    assert len({chunk.chunk_id for chunk in first.chunks}) == first.chunk_count
    assert [chunk.chunk_index for chunk in first.chunks] == list(
        range(first.chunk_count)
    )


def test_split_text_adds_overlap_without_exceeding_limit() -> None:
    config = ChunkingConfig(max_chunk_chars=90, overlap_chars=20)
    text = " ".join(f"token-{index}" for index in range(40))

    chunks = split_text(text, config)

    assert len(chunks) > 1
    assert all(len(chunk) <= 90 for chunk in chunks)
    previous_words = set(chunks[0].split())
    assert previous_words.intersection(chunks[1].split())


def test_split_text_falls_back_to_character_boundaries() -> None:
    text = "检索增强生成" * 40

    chunks = split_text(
        text,
        ChunkingConfig(max_chunk_chars=64, overlap_chars=0),
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 64 for chunk in chunks)
    assert "".join(chunks) == text


@pytest.mark.parametrize(
    ("max_chars", "overlap"),
    [(63, 10), (100, -1), (100, 51), (100, 100)],
)
def test_invalid_chunking_config_is_rejected(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(max_chunk_chars=max_chars, overlap_chars=overlap)


def test_blank_document_returns_warning() -> None:
    document = ParsedDocument(
        document_id="empty",
        source_file="empty.pdf",
        sha256="b" * 64,
        title="Empty",
        author=None,
        page_count=1,
        pages=(ParsedPage(page_number=1, text="", word_count=0),),
    )

    result = chunk_document(document)

    assert result.chunk_count == 0
    assert "no extractable text" in result.warnings[-1]


def test_chunking_cli_runs_pdf_to_json(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    output_path = tmp_path / "paper.chunks.json"
    canvas = Canvas(str(pdf_path))
    canvas.setTitle("CLI Paper")
    canvas.drawString(72, 760, "1 Introduction")
    canvas.drawString(72, 730, "This paper evaluates a retrieval pipeline.")
    canvas.save()

    exit_code = main(
        [
            str(pdf_path),
            "--output",
            str(output_path),
            "--max-chars",
            "100",
            "--overlap-chars",
            "20",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["title"] == "CLI Paper"
    assert payload["chunk_count"] == 1
    assert payload["chunks"][0]["section"] == "1 Introduction"
    assert payload["chunks"][0]["page_number"] == 1
