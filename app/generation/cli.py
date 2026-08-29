"""Command-line interface for retrieval-augmented DeepSeek answers."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from app.generation.grounded import CitationValidationError, GroundedAnswerGenerator
from app.generation.llm import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekClient,
    LLMError,
)
from app.reranking import (
    DEFAULT_RERANKER_MODEL,
    RerankingRetriever,
    SentenceTransformersCrossEncoder,
)
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.cli import DEFAULT_DB_PATH
from app.retrieval.dense import DEFAULT_COLLECTION, DenseRetrievalError, DenseRetriever
from app.retrieval.embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedProvider
from app.retrieval.hybrid import HybridRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer a question from indexed papers with verified citations."
    )
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--document-id")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--strategy",
        choices=("dense", "bm25", "hybrid", "hybrid_rerank"),
        default="hybrid",
    )
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--deepseek-base-url", default=DEFAULT_DEEPSEEK_BASE_URL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    qdrant: QdrantClient | None = None
    llm: DeepSeekClient | None = None
    try:
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
            if args.strategy == "dense":
                retriever = dense
            else:
                hybrid = HybridRetriever(dense, sparse)
                retriever = (
                    RerankingRetriever(
                        hybrid,
                        SentenceTransformersCrossEncoder(args.reranker_model),
                    )
                    if args.strategy == "hybrid_rerank"
                    else hybrid
                )
        search = retriever.search(
            args.query,
            top_k=args.top_k,
            document_id=args.document_id,
        )
        llm = DeepSeekClient.from_env(
            model_name=args.llm_model,
            base_url=args.deepseek_base_url,
        )
        answer = GroundedAnswerGenerator(llm).generate(args.query, search.hits)
    except (
        CitationValidationError,
        DenseRetrievalError,
        LLMError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if llm is not None:
            llm.close()
        if qdrant is not None:
            qdrant.close()

    print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
