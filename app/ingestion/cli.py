"""Command-line interface for parsing a local paper PDF."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from app.ingestion.pdf_parser import PDFParseError, parse_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse a text-based paper PDF into page-addressable JSON."
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write JSON to this path instead of standard output",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of indented JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        document = parse_pdf(args.pdf)
    except PDFParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    indent = None if args.compact else 2
    payload = json.dumps(document.to_dict(), ensure_ascii=False, indent=indent) + "\n"

    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
