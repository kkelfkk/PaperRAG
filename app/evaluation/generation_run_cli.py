"""Generate and save PaperRAG outputs for a frozen evaluation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from app.evaluation.generation import (
    GenerationPrediction,
    GenerationPredictionSet,
    load_generation_dataset,
)
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
from app.retrieval.decomposition import DecomposedRetriever, load_query_decomposer
from app.retrieval.dense import DenseRetrievalError, DenseRetriever
from app.retrieval.embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedProvider
from app.retrieval.hybrid import HybridRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed PaperRAG configuration and save generation outputs."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=Path("storage/qdrant_eval"))
    parser.add_argument("--collection", default="paperrag_eval_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--strategy",
        choices=("hybrid_rerank", "decomposed_hybrid_rerank"),
        default="decomposed_hybrid_rerank",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("configs/eval_corpus.json"),
    )
    parser.add_argument("--llm-model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--deepseek-base-url", default=DEFAULT_DEEPSEEK_BASE_URL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.top_k <= 0:
        print("error: top_k must be positive", file=sys.stderr)
        return 2

    qdrant: QdrantClient | None = None
    llm: DeepSeekClient | None = None
    try:
        dataset = load_generation_dataset(args.dataset)
        qdrant = QdrantClient(path=str(args.db_path))
        if not qdrant.collection_exists(args.collection):
            raise ValueError(f"Qdrant collection does not exist: {args.collection}")
        dense = DenseRetriever(
            qdrant,
            FastEmbedProvider(args.embedding_model),
            args.collection,
        )
        hybrid = HybridRetriever(dense, BM25Retriever(qdrant, args.collection))
        reranked = RerankingRetriever(
            hybrid,
            SentenceTransformersCrossEncoder(args.reranker_model),
        )
        retriever = (
            DecomposedRetriever(
                reranked,
                load_query_decomposer(args.corpus_manifest),
            )
            if args.strategy == "decomposed_hybrid_rerank"
            else reranked
        )
        llm = DeepSeekClient.from_env(
            model_name=args.llm_model,
            base_url=args.deepseek_base_url,
        )
        generator = GroundedAnswerGenerator(llm)
        predictions: list[GenerationPrediction] = []
        retrieval_configuration = ""

        for query in dataset.queries:
            search = retriever.search(query.question, top_k=args.top_k)
            if not retrieval_configuration:
                retrieval_configuration = search.embedding_model
            elif search.embedding_model != retrieval_configuration:
                raise ValueError("retrieval configuration changed during generation")
            answer = generator.generate(query.question, search.hits)
            used_hits = search.hits[: answer.retrieved_count]
            predictions.append(
                GenerationPrediction(
                    query_id=query.query_id,
                    answer=answer.answer,
                    abstained=answer.abstained,
                    cited_chunk_ids=[citation.chunk_id for citation in answer.citations],
                    retrieved_chunk_ids=[hit.chunk_id for hit in used_hits],
                    evidence={hit.chunk_id: hit.text for hit in used_hits},
                )
            )

        payload = GenerationPredictionSet(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            system_name=(
                f"{args.strategy};top_k={args.top_k};{retrieval_configuration}"
            ),
            model=llm.model_name,
            predictions=predictions,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
