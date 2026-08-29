"""Tests for the reproducible evaluation corpus workflow."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from app.retrieval.models import SearchHit, SearchResponse
from scripts.eval_corpus import (
    CorpusManifest,
    PaperSpec,
    build_worksheet,
    download_paper,
    load_manifest,
    load_question_seeds,
)


def test_committed_manifest_and_question_seeds_are_consistent() -> None:
    manifest = load_manifest()
    seeds = load_question_seeds()

    assert manifest.corpus_id == seeds.corpus_id
    assert len(manifest.papers) == 4
    assert len(seeds.questions) == 30
    known_papers = {paper.arxiv_id for paper in manifest.papers}
    assert all(
        set(question.target_papers).issubset(known_papers)
        for question in seeds.questions
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://arxiv.org/pdf/2005.11401v4.pdf",
        "https://example.com/pdf/2005.11401v4.pdf",
        "https://arxiv.org/pdf/other.pdf",
    ],
)
def test_manifest_rejects_unpinned_or_untrusted_urls(url: str) -> None:
    with pytest.raises(ValidationError, match="paper URL"):
        PaperSpec(
            arxiv_id="2005.11401v4",
            title="Paper",
            short_name="RAG",
            url=url,
            filename="paper.pdf",
            sha256="0" * 64,
        )


def test_manifest_rejects_duplicate_papers() -> None:
    paper = {
        "arxiv_id": "2005.11401v4",
        "title": "Paper",
        "short_name": "RAG",
        "url": "https://arxiv.org/pdf/2005.11401v4.pdf",
        "filename": "paper.pdf",
        "sha256": "0" * 64,
    }
    with pytest.raises(ValidationError, match="unique"):
        CorpusManifest(
            corpus_id="corpus",
            description="description",
            papers=[paper, paper],
        )


def test_download_paper_validates_and_hashes_pdf(tmp_path: Path) -> None:
    content = b"%PDF-1.7\nmock paper"
    paper = load_manifest().papers[0].model_copy(
        update={"sha256": hashlib.sha256(content).hexdigest()}
    )
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=content,
                request=request,
            )
        )
    )

    first = download_paper(paper, tmp_path, http_client)
    second = download_paper(paper, tmp_path, http_client)

    assert first["downloaded"] is True
    assert second["downloaded"] is False
    assert first["sha256"] == second["sha256"]
    assert (tmp_path / paper.filename).read_bytes().startswith(b"%PDF-")
    http_client.close()


def test_download_rejects_non_pdf_and_removes_partial_file(tmp_path: Path) -> None:
    paper = load_manifest().papers[0]
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=b"not a PDF",
                request=request,
            )
        )
    )

    with pytest.raises(ValueError, match="did not return a PDF"):
        download_paper(paper, tmp_path, http_client)

    assert not list(tmp_path.iterdir())
    http_client.close()


def test_download_rejects_pdf_with_wrong_checksum(tmp_path: Path) -> None:
    paper = load_manifest().papers[0]
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=b"%PDF-1.7\ntampered",
                request=request,
            )
        )
    )

    with pytest.raises(ValueError, match="checksum"):
        download_paper(paper, tmp_path, http_client)

    assert not list(tmp_path.iterdir())
    http_client.close()


class FakeSearcher:
    def search(self, query: str, *, top_k: int = 5) -> SearchResponse:
        return SearchResponse(
            query=query,
            collection_name="papers",
            embedding_model="test-model",
            hits=(
                SearchHit(
                    rank=1,
                    score=0.9,
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    source_file="paper.pdf",
                    title="Paper",
                    page_number=4,
                    chunk_index=0,
                    section="Methods",
                    text="evidence " * 100,
                ),
            )[:top_k],
        )


def test_worksheet_contains_candidates_but_no_fake_labels() -> None:
    seeds = load_question_seeds()

    worksheet = build_worksheet(
        seeds,
        FakeSearcher(),
        corpus_fingerprint="fingerprint",
        top_k=20,
    )

    assert len(worksheet["questions"]) == 30
    first = worksheet["questions"][0]
    assert first["relevant_chunk_ids"] == []
    assert first["candidates"][0]["chunk_id"] == "chunk-1"
    assert len(first["candidates"][0]["text_preview"]) == 600
