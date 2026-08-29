"""Command-line interface for parsing and chunking a paper PDF."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from app.chunking.models import ChunkingConfig
from app.chunking.recursive import chunk_document
from app.ingestion.pdf_parser import PDFParseError, parse_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse and split a paper PDF into retrieval-ready JSON chunks."
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write chunk JSON to this path instead of standard output",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Maximum characters per chunk (default: 1200)",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=200,
        help="Maximum repeated context between chunks (default: 200)",
    )
    parser.add_argument("--compact", action="store_true", help="Write compact JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = ChunkingConfig(
            max_chunk_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
        result = chunk_document(parse_pdf(args.pdf), config)
    except (PDFParseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    indent = None if args.compact else 2
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=indent) + "\n"
    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
