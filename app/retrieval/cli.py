"""Command-line interface for local Qdrant indexing and dense search."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import QdrantClient

from app.chunking.models import ChunkingConfig
from app.chunking.recursive import chunk_document
from app.ingestion.pdf_parser import PDFParseError, parse_pdf
from app.retrieval.dense import DEFAULT_COLLECTION, DenseRetrievalError, DenseRetriever
from app.retrieval.embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedProvider

DEFAULT_DB_PATH = Path("storage/qdrant")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index paper chunks and search them with local Qdrant."
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Parse and index one PDF")
    index_parser.add_argument("pdf", type=Path)
    index_parser.add_argument("--max-chars", type=int, default=1200)
    index_parser.add_argument("--overlap-chars", type=int, default=200)
    index_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the whole collection before indexing",
    )

    search_parser = subparsers.add_parser("search", help="Search indexed chunks")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--document-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client: QdrantClient | None = None
    try:
        args.db_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(args.db_path))
        if args.command == "search" and not client.collection_exists(args.collection):
            raise DenseRetrievalError(
                f"Qdrant collection does not exist: {args.collection}. "
                "Run the index command first."
            )
        chunks = None
        if args.command == "index":
            config = ChunkingConfig(
                max_chunk_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
            )
            chunks = chunk_document(parse_pdf(args.pdf), config).chunks
        retriever = DenseRetriever(
            client=client,
            embedder=FastEmbedProvider(args.model),
            collection_name=args.collection,
        )
        if args.command == "index":
            result = retriever.index_document(
                chunks or (),
                recreate=args.recreate,
            )
        else:
            result = retriever.search(
                args.query,
                top_k=args.top_k,
                document_id=args.document_id,
            )
    except (PDFParseError, DenseRetrievalError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client.close()

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
