"""Application service that composes ingestion, retrieval, and generation."""

from __future__ import annotations

import os
import threading
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from app.chunking.models import ChunkingConfig
from app.chunking.recursive import chunk_document
from app.generation.grounded import GroundedAnswerGenerator
from app.generation.llm import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekClient,
    LLMError,
)
from app.generation.models import GroundedAnswer
from app.ingestion.pdf_parser import parse_pdf
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DEFAULT_COLLECTION, DenseRetriever
from app.retrieval.embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedProvider
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.models import IndexReport, SearchResponse


class PaperRAGService:
    """Thread-safe facade for the local baseline pipeline."""

    def __init__(
        self,
        retriever: DenseRetriever,
        answer_generator: GroundedAnswerGenerator | None = None,
    ) -> None:
        self.retriever = retriever
        self.sparse_retriever = BM25Retriever(
            retriever.client,
            retriever.collection_name,
        )
        self.hybrid_retriever = HybridRetriever(
            retriever,
            self.sparse_retriever,
        )
        self.answer_generator = answer_generator
        self._lock = threading.RLock()

    def index_pdf(
        self,
        pdf_path: Path,
        *,
        source_name: str,
        max_chunk_chars: int = 1200,
        overlap_chars: int = 200,
        recreate: bool = False,
    ) -> IndexReport:
        safe_name = Path(source_name.replace("\\", "/")).name or "uploaded.pdf"
        with self._lock:
            document = parse_pdf(pdf_path)
            replacement_title = (
                Path(safe_name).stem
                if document.title == pdf_path.stem
                else document.title
            )
            document = replace(
                document,
                source_file=safe_name,
                title=replacement_title,
            )
            chunks = chunk_document(
                document,
                ChunkingConfig(
                    max_chunk_chars=max_chunk_chars,
                    overlap_chars=overlap_chars,
                ),
            ).chunks
            return self.retriever.index_document(chunks, recreate=recreate)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
        strategy: str = "hybrid",
    ) -> SearchResponse:
        with self._lock:
            searcher = self._searcher(strategy)
            return searcher.search(
                query,
                top_k=top_k,
                document_id=document_id,
            )

    def ask(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
        strategy: str = "hybrid",
    ) -> GroundedAnswer:
        if self.answer_generator is None:
            raise LLMError(
                "DeepSeek is not configured. Add a newly created "
                "DEEPSEEK_API_KEY to the local .env file and restart the API."
            )
        with self._lock:
            search = self._searcher(strategy).search(
                query,
                top_k=top_k,
                document_id=document_id,
            )
            return self.answer_generator.generate(query, search.hits)

    def _searcher(
        self,
        strategy: str,
    ) -> DenseRetriever | BM25Retriever | HybridRetriever:
        if strategy == "dense":
            return self.retriever
        if strategy == "bm25":
            return self.sparse_retriever
        if strategy == "hybrid":
            return self.hybrid_retriever
        raise ValueError(f"unknown retrieval strategy: {strategy}")

    def close(self) -> None:
        llm = getattr(self.answer_generator, "llm", None)
        if isinstance(llm, DeepSeekClient):
            llm.close()
        self.retriever.client.close()


def create_default_service() -> PaperRAGService:
    """Create the lazily loaded local service from environment configuration."""

    load_dotenv()
    db_path = Path(os.getenv("QDRANT_PATH", "storage/qdrant"))
    db_path.mkdir(parents=True, exist_ok=True)
    qdrant = QdrantClient(path=str(db_path))
    retriever = DenseRetriever(
        client=qdrant,
        embedder=FastEmbedProvider(
            os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        ),
        collection_name=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
    )

    answer_generator = None
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if api_key:
        llm = DeepSeekClient(
            api_key,
            model_name=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        )
        answer_generator = GroundedAnswerGenerator(llm)
    return PaperRAGService(retriever, answer_generator)
