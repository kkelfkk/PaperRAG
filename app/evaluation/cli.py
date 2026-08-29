"""Command-line entry point for retrieval dataset validation and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import QdrantClient

from app.evaluation.dataset import load_dataset
from app.evaluation.runner import evaluate_retriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.cli import DEFAULT_DB_PATH
from app.retrieval.dense import DEFAULT_COLLECTION, DenseRetrievalError, DenseRetriever
from app.retrieval.embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedProvider
from app.retrieval.hybrid import HybridRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or run a labeled PaperRAG retrieval evaluation set."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--strategy",
        choices=("dense", "bm25", "hybrid"),
        default="dense",
        help="Retrieval strategy to evaluate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    qdrant: QdrantClient | None = None
    try:
        dataset = load_dataset(args.dataset)
        if args.validate_only:
            payload = {
                "valid": True,
                "dataset_name": dataset.name,
                "version": dataset.version,
                "query_count": len(dataset.queries),
            }
        else:
            qdrant = QdrantClient(path=str(args.db_path))
            sparse = BM25Retriever(qdrant, args.collection)
            if args.strategy == "bm25":
                retriever = sparse
            else:
                dense = DenseRetriever(
                    client=qdrant,
                    embedder=FastEmbedProvider(args.embedding_model),
                    collection_name=args.collection,
                )
                retriever = (
                    dense
                    if args.strategy == "dense"
                    else HybridRetriever(dense, sparse)
                )
            payload = evaluate_retriever(
                dataset,
                retriever,
                cutoffs=args.cutoffs,
            ).to_dict()
    except (DenseRetrievalError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if qdrant is not None:
            qdrant.close()

    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
