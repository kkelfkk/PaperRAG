"""Download, index, and prepare annotation candidates for the eval corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, model_validator
from qdrant_client import QdrantClient

from app.chunking.models import ChunkingConfig
from app.chunking.recursive import chunk_document
from app.ingestion.pdf_parser import parse_pdf
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.cli import DEFAULT_DB_PATH
from app.retrieval.dense import DEFAULT_COLLECTION, DenseRetriever
from app.retrieval.embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedProvider
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.models import SearchResponse

DEFAULT_MANIFEST = Path("configs/eval_corpus.json")
DEFAULT_QUESTIONS = Path("data/eval/question_seeds.json")
DEFAULT_PAPER_DIR = Path("data/papers/eval")
DEFAULT_LOCK_PATH = DEFAULT_PAPER_DIR / "corpus.lock.json"
DEFAULT_WORKSHEET = Path("data/eval/work/retrieval_candidates.json")
MAX_PDF_BYTES = 50 * 1024 * 1024
ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}v\d+$")


class PaperSpec(BaseModel):
    arxiv_id: str = Field(pattern=ARXIV_ID_PATTERN.pattern)
    title: str = Field(min_length=1, max_length=500)
    short_name: str = Field(min_length=1, max_length=100)
    url: str
    filename: str = Field(pattern=r"^[A-Za-z0-9._-]+\.pdf$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_source(self) -> PaperSpec:
        parsed = urlparse(self.url)
        expected_path = f"/pdf/{self.arxiv_id}.pdf"
        if parsed.scheme != "https" or parsed.hostname != "arxiv.org":
            raise ValueError("paper URL must use https://arxiv.org")
        if parsed.path != expected_path or parsed.query or parsed.fragment:
            raise ValueError(f"paper URL must end with {expected_path}")
        return self


class CorpusManifest(BaseModel):
    corpus_id: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1000)
    papers: list[PaperSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_papers(self) -> CorpusManifest:
        for values, label in (
            ([paper.arxiv_id for paper in self.papers], "arxiv_id"),
            ([paper.filename for paper in self.papers], "filename"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"paper {label} values must be unique")
        return self


class QuestionSeed(BaseModel):
    query_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)
    question_type: str = Field(
        pattern=r"^(factual|terminology|comparison|multi_evidence)$"
    )
    target_papers: list[str] = Field(min_length=1)


class QuestionSeeds(BaseModel):
    corpus_id: str = Field(min_length=1, max_length=200)
    questions: list[QuestionSeed] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_questions(self) -> QuestionSeeds:
        query_ids = [question.query_id for question in self.questions]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("question query_id values must be unique")
        return self


class Searcher(Protocol):
    def search(self, query: str, *, top_k: int = 5) -> SearchResponse: ...


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def load_manifest(path: Path = DEFAULT_MANIFEST) -> CorpusManifest:
    return CorpusManifest.model_validate(_load_json(path))


def load_question_seeds(path: Path = DEFAULT_QUESTIONS) -> QuestionSeeds:
    return QuestionSeeds.model_validate(_load_json(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_pdf(path: Path, expected_sha256: str | None = None) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"downloaded PDF is empty: {path}")
    if path.stat().st_size > MAX_PDF_BYTES:
        raise ValueError(f"downloaded PDF exceeds 50 MB: {path}")
    with path.open("rb") as file:
        if file.read(5) != b"%PDF-":
            raise ValueError(f"download did not return a PDF: {path}")
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError(f"PDF checksum does not match manifest: {path}")


def download_paper(
    paper: PaperSpec,
    output_dir: Path,
    client: httpx.Client,
    *,
    force: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / paper.filename
    if destination.exists() and not force:
        _validate_pdf(destination, paper.sha256)
        return {
            "arxiv_id": paper.arxiv_id,
            "filename": paper.filename,
            "sha256": _sha256(destination),
            "bytes": destination.stat().st_size,
            "downloaded": False,
        }

    temporary = destination.with_suffix(".pdf.part")
    temporary.unlink(missing_ok=True)
    received = 0
    try:
        with client.stream("GET", paper.url) as response:
            response.raise_for_status()
            with temporary.open("wb") as file:
                for block in response.iter_bytes(chunk_size=1024 * 1024):
                    received += len(block)
                    if received > MAX_PDF_BYTES:
                        raise ValueError(f"PDF exceeds 50 MB: {paper.arxiv_id}")
                    file.write(block)
        _validate_pdf(temporary, paper.sha256)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "arxiv_id": paper.arxiv_id,
        "filename": paper.filename,
        "sha256": _sha256(destination),
        "bytes": destination.stat().st_size,
        "downloaded": True,
    }


def download_corpus(
    manifest: CorpusManifest,
    output_dir: Path = DEFAULT_PAPER_DIR,
    *,
    force: bool = False,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    owns_client = client is None
    http_client = client or httpx.Client(
        follow_redirects=True,
        timeout=60.0,
        headers={"User-Agent": "PaperRAG/0.1 evaluation corpus"},
    )
    try:
        return [
            download_paper(paper, output_dir, http_client, force=force)
            for paper in manifest.papers
        ]
    finally:
        if owns_client:
            http_client.close()


def index_corpus(
    manifest: CorpusManifest,
    paper_dir: Path = DEFAULT_PAPER_DIR,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    collection: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    recreate: bool = False,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> dict[str, Any]:
    db_path.mkdir(parents=True, exist_ok=True)
    qdrant = QdrantClient(path=str(db_path))
    chunking_config = ChunkingConfig()
    try:
        retriever = DenseRetriever(
            qdrant,
            FastEmbedProvider(embedding_model),
            collection,
        )
        indexed: list[dict[str, Any]] = []
        for index, paper in enumerate(manifest.papers):
            path = paper_dir / paper.filename
            _validate_pdf(path, paper.sha256)
            parsed = replace(
                parse_pdf(path),
                source_file=paper.filename,
                title=paper.title,
            )
            chunks = chunk_document(parsed, chunking_config).chunks
            report = retriever.index_document(
                chunks,
                recreate=recreate and index == 0,
            )
            indexed.append(
                {
                    "arxiv_id": paper.arxiv_id,
                    "filename": paper.filename,
                    "sha256": _sha256(path),
                    "document_id": report.document_id,
                    "chunk_count": report.indexed_chunks,
                }
            )
    finally:
        qdrant.close()

    fingerprint_input = {
        "papers": indexed,
        "embedding_model": embedding_model,
        "chunking_config": asdict(chunking_config),
    }
    fingerprint_payload = json.dumps(fingerprint_input, sort_keys=True).encode()
    lock = {
        "corpus_id": manifest.corpus_id,
        "corpus_fingerprint": hashlib.sha256(fingerprint_payload).hexdigest(),
        "collection_name": collection,
        "embedding_model": embedding_model,
        "chunking_config": asdict(chunking_config),
        "papers": indexed,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return lock


def build_worksheet(
    seeds: QuestionSeeds,
    searcher: Searcher,
    *,
    corpus_fingerprint: str,
    top_k: int = 20,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    questions: list[dict[str, Any]] = []
    for seed in seeds.questions:
        response = searcher.search(seed.question, top_k=top_k)
        questions.append(
            {
                **seed.model_dump(),
                "relevant_chunk_ids": [],
                "annotation_notes": "",
                "candidates": [
                    {
                        "rank": hit.rank,
                        "score": hit.score,
                        "chunk_id": hit.chunk_id,
                        "document_id": hit.document_id,
                        "source_file": hit.source_file,
                        "title": hit.title,
                        "page_number": hit.page_number,
                        "section": hit.section,
                        "text_preview": hit.text[:600],
                    }
                    for hit in response.hits
                ],
            }
        )
    return {
        "corpus_id": seeds.corpus_id,
        "corpus_fingerprint": corpus_fingerprint,
        "instructions": (
            "Inspect the cited PDF pages, copy every relevant chunk_id into "
            "relevant_chunk_ids, and explain borderline decisions. Empty labels "
            "are intentionally not valid evaluation data."
        ),
        "questions": questions,
    }


def create_worksheet(
    seeds: QuestionSeeds,
    *,
    lock_path: Path = DEFAULT_LOCK_PATH,
    output_path: Path = DEFAULT_WORKSHEET,
    db_path: Path = DEFAULT_DB_PATH,
    collection: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 20,
) -> dict[str, Any]:
    lock = _load_json(lock_path)
    if lock.get("corpus_id") != seeds.corpus_id:
        raise ValueError("question seeds and corpus lock use different corpus IDs")
    qdrant = QdrantClient(path=str(db_path))
    try:
        dense = DenseRetriever(qdrant, FastEmbedProvider(embedding_model), collection)
        searcher = HybridRetriever(dense, BM25Retriever(qdrant, collection))
        worksheet = build_worksheet(
            seeds,
            searcher,
            corpus_fingerprint=str(lock["corpus_fingerprint"]),
            top_k=top_k,
        )
    finally:
        qdrant.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(worksheet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return worksheet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--force", action="store_true")

    index = subparsers.add_parser("index")
    index.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    index.add_argument("--collection", default=DEFAULT_COLLECTION)
    index.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    index.add_argument("--recreate", action="store_true")

    worksheet = subparsers.add_parser("worksheet")
    worksheet.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    worksheet.add_argument("--output", type=Path, default=DEFAULT_WORKSHEET)
    worksheet.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    worksheet.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    worksheet.add_argument("--collection", default=DEFAULT_COLLECTION)
    worksheet.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    worksheet.add_argument("--top-k", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "download":
            result: Any = download_corpus(
                manifest,
                args.paper_dir,
                force=args.force,
            )
        elif args.command == "index":
            result = index_corpus(
                manifest,
                args.paper_dir,
                db_path=args.db_path,
                collection=args.collection,
                embedding_model=args.embedding_model,
                recreate=args.recreate,
            )
        else:
            seeds = load_question_seeds(args.questions)
            worksheet_result = create_worksheet(
                seeds,
                lock_path=args.lock_path,
                output_path=args.output,
                db_path=args.db_path,
                collection=args.collection,
                embedding_model=args.embedding_model,
                top_k=args.top_k,
            )
            result = {
                "output": str(args.output),
                "corpus_id": worksheet_result["corpus_id"],
                "corpus_fingerprint": worksheet_result["corpus_fingerprint"],
                "question_count": len(worksheet_result["questions"]),
                "candidate_count": sum(
                    len(question["candidates"])
                    for question in worksheet_result["questions"]
                ),
            }
    except (httpx.HTTPError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
