"""Tests for page-addressable PDF ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from reportlab.pdfgen.canvas import Canvas

from app.ingestion.cli import main
from app.ingestion.pdf_parser import PDFParseError, detect_headings, parse_pdf


def _make_pdf(path: Path, *, include_blank_page: bool = False) -> None:
    canvas = Canvas(str(path))
    canvas.setTitle("A Test Paper")
    canvas.setAuthor("PaperRAG Tests")
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawString(72, 760, "1 Introduction")
    canvas.setFont("Helvetica", 11)
    canvas.drawString(72, 730, "Retrieval-augmented generation uses external evidence.")
    canvas.showPage()

    if not include_blank_page:
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawString(72, 760, "2.1 Retrieval Pipeline")
        canvas.setFont("Helvetica", 11)
        canvas.drawString(72, 730, "The retriever returns relevant passages.")
    canvas.showPage()
    canvas.save()


def test_parse_pdf_preserves_metadata_pages_and_headings(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path)

    document = parse_pdf(pdf_path)

    assert document.title == "A Test Paper"
    assert document.author == "PaperRAG Tests"
    assert document.source_file == "paper.pdf"
    assert document.page_count == 2
    assert len(document.sha256) == 64
    assert document.document_id == document.sha256[:16]
    assert document.pages[0].page_number == 1
    assert "external evidence" in document.pages[0].text
    assert document.pages[0].headings[0].text == "1 Introduction"
    assert document.pages[1].headings[0].level == 2
    assert not document.warnings


def test_parse_pdf_retains_blank_page_and_emits_warning(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank-page.pdf"
    _make_pdf(pdf_path, include_blank_page=True)

    document = parse_pdf(pdf_path)

    assert document.page_count == 2
    assert document.pages[1].page_number == 2
    assert document.pages[1].text == ""
    assert document.warnings == (
        "Page 2 contains no extractable text; OCR may be required.",
    )


def test_parser_rejects_missing_and_fake_pdf(tmp_path: Path) -> None:
    with pytest.raises(PDFParseError, match="does not exist"):
        parse_pdf(tmp_path / "missing.pdf")

    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(PDFParseError, match="valid PDF header"):
        parse_pdf(fake_pdf)


def test_heading_detection_is_conservative() -> None:
    headings = detect_headings(
        "Abstract\nOrdinary explanatory sentence.\n3.2 Hybrid Search\nREFERENCES",
        page_number=4,
    )

    assert [(item.text, item.level) for item in headings] == [
        ("Abstract", 1),
        ("3.2 Hybrid Search", 2),
        ("REFERENCES", 1),
    ]


def test_cli_writes_utf8_json(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    output_path = tmp_path / "nested" / "paper.json"
    _make_pdf(pdf_path)

    exit_code = main([str(pdf_path), "--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["page_count"] == 2
    assert payload["pages"][0]["page_number"] == 1
