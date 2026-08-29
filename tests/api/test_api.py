"""HTTP contract tests for the FastAPI application."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api.main as api_main
from app.api.main import create_app, get_service
from app.generation.llm import LLMError
from app.generation.models import Citation, GroundedAnswer
from app.retrieval.models import IndexReport, SearchHit, SearchResponse


class FakeService:
    def __init__(self) -> None:
        self.upload_path: Path | None = None
        self.upload_exists_during_call = False
        self.upload_header = b""
        self.source_name = ""
        self.ask_error: Exception | None = None
        self.search_options: dict[str, object] = {}
        self.ask_options: dict[str, object] = {}

    def index_pdf(
        self, pdf_path: Path, *, source_name: str, **kwargs: object
    ) -> IndexReport:
        del kwargs
        self.upload_path = pdf_path
        self.upload_exists_during_call = pdf_path.exists()
        self.upload_header = pdf_path.read_bytes()[:5]
        self.source_name = source_name
        return IndexReport(
            collection_name="paperrag_dense",
            document_id="doc-1",
            indexed_chunks=3,
            vector_size=384,
            embedding_model="test/embedder",
        )

    def search(self, query: str, **kwargs: object) -> SearchResponse:
        self.search_options = kwargs
        return SearchResponse(
            query=query,
            collection_name="paperrag_dense",
            embedding_model="test/embedder",
            hits=(
                SearchHit(
                    rank=1,
                    score=0.91,
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    source_file="paper.pdf",
                    title="Paper",
                    page_number=4,
                    chunk_index=0,
                    section="Introduction",
                    text="Evidence passage.",
                ),
            ),
        )

    def ask(self, query: str, **kwargs: object) -> GroundedAnswer:
        self.ask_options = kwargs
        if self.ask_error is not None:
            raise self.ask_error
        return GroundedAnswer(
            query=query,
            answer="Grounded answer. [S1]",
            abstained=False,
            model="test/llm",
            retrieved_count=1,
            citations=(
                Citation(
                    source_id="S1",
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    source_file="paper.pdf",
                    title="Paper",
                    page_number=4,
                    section="Introduction",
                ),
            ),
        )


@pytest.fixture
def api_client() -> tuple[TestClient, FakeService]:
    application = create_app()
    service = FakeService()
    application.dependency_overrides[get_service] = lambda: service
    with TestClient(application) as client:
        yield client, service


def test_health_does_not_load_pipeline() -> None:
    application = create_app()
    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "paperrag",
        "version": "0.1.0",
    }


def test_upload_indexes_pdf_and_removes_temporary_file(
    api_client: tuple[TestClient, FakeService],
) -> None:
    client, service = api_client

    response = client.post(
        "/v1/documents/index",
        files={"file": ("paper.pdf", b"%PDF-test-content", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["indexed_chunks"] == 3
    assert service.upload_exists_during_call
    assert service.upload_header == b"%PDF-"
    assert service.source_name == "paper.pdf"
    assert service.upload_path is not None
    assert not service.upload_path.exists()


def test_upload_rejects_non_pdf(api_client: tuple[TestClient, FakeService]) -> None:
    client, _ = api_client

    response = client.post(
        "/v1/documents/index",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415


def test_upload_enforces_size_limit(
    api_client: tuple[TestClient, FakeService], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, service = api_client
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 5)

    response = client.post(
        "/v1/documents/index",
        files={"file": ("paper.pdf", b"%PDF-too-large", "application/pdf")},
    )

    assert response.status_code == 413
    assert service.upload_path is None


def test_search_returns_ranked_source_metadata(
    api_client: tuple[TestClient, FakeService],
) -> None:
    client, service = api_client

    response = client.post(
        "/v1/search",
        json={"query": "evidence", "top_k": 3, "strategy": "bm25"},
    )

    assert response.status_code == 200
    assert response.json()["hits"][0]["page_number"] == 4
    assert response.json()["hits"][0]["score"] == 0.91
    assert service.search_options["strategy"] == "bm25"


def test_ask_returns_answer_and_citations(
    api_client: tuple[TestClient, FakeService],
) -> None:
    client, service = api_client

    response = client.post("/v1/ask", json={"query": "question"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Grounded answer. [S1]"
    assert response.json()["citations"][0]["page_number"] == 4
    assert service.ask_options["strategy"] == "hybrid"


def test_request_validation_rejects_unknown_strategy(
    api_client: tuple[TestClient, FakeService],
) -> None:
    client, _ = api_client

    response = client.post(
        "/v1/search",
        json={"query": "question", "strategy": "unknown"},
    )

    assert response.status_code == 422


def test_api_accepts_hybrid_rerank_strategy(
    api_client: tuple[TestClient, FakeService],
) -> None:
    client, service = api_client

    response = client.post(
        "/v1/search",
        json={"query": "question", "strategy": "hybrid_rerank"},
    )

    assert response.status_code == 200
    assert service.search_options["strategy"] == "hybrid_rerank"


def test_ask_without_deepseek_configuration_returns_503(
    api_client: tuple[TestClient, FakeService],
) -> None:
    client, service = api_client
    service.ask_error = LLMError("DeepSeek is not configured")

    response = client.post("/v1/ask", json={"query": "question"})

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_request_validation_rejects_invalid_top_k(
    api_client: tuple[TestClient, FakeService],
) -> None:
    client, _ = api_client

    response = client.post("/v1/search", json={"query": "question", "top_k": 0})

    assert response.status_code == 422
