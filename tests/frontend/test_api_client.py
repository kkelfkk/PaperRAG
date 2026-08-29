"""Tests for the Streamlit frontend's local FastAPI client."""

from __future__ import annotations

import json

import httpx
import pytest

from frontend.api_client import PaperRAGAPIClient, PaperRAGAPIError


def _client(handler: httpx.MockTransport) -> tuple[PaperRAGAPIClient, httpx.Client]:
    http_client = httpx.Client(transport=handler)
    return PaperRAGAPIClient(http_client=http_client), http_client


def test_health_and_search_use_expected_endpoints_and_filters() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "version": "0.1.0"})
        return httpx.Response(200, json={"query": "q", "hits": []})

    client, http_client = _client(httpx.MockTransport(handler))

    assert client.health()["status"] == "ok"
    result = client.search(
        "q",
        strategy="hybrid_rerank",
        document_id="doc-1",
        section="Methods",
        page_from=4,
        page_to=9,
    )

    assert result["hits"] == []
    payload = json.loads(requests[1].content)
    assert requests[1].url.path == "/v1/search"
    assert payload == {
        "query": "q",
        "top_k": 5,
        "strategy": "hybrid_rerank",
        "document_id": "doc-1",
        "section": "Methods",
        "page_from": 4,
        "page_to": 9,
    }
    http_client.close()


def test_ask_omits_empty_optional_filters() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok", "citations": []})

    client, http_client = _client(httpx.MockTransport(handler))

    client.ask("question", document_id=None, section="")

    assert captured == {"query": "question", "top_k": 5, "strategy": "hybrid"}
    http_client.close()


def test_index_pdf_sends_multipart_file_and_parameters() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={"document_id": "doc-1", "indexed_chunks": 3},
        )

    client, http_client = _client(httpx.MockTransport(handler))

    report = client.index_pdf(
        "paper.pdf",
        b"%PDF-test",
        recreate=True,
        max_chunk_chars=900,
        overlap_chars=100,
    )

    assert report["indexed_chunks"] == 3
    assert captured["path"] == "/v1/documents/index"
    assert str(captured["content_type"]).startswith("multipart/form-data;")
    body = captured["body"]
    assert isinstance(body, bytes)
    assert b'filename="paper.pdf"' in body
    assert b"%PDF-test" in body
    assert b"900" in body
    http_client.close()


def test_api_error_uses_safe_detail_message() -> None:
    client, http_client = _client(
        httpx.MockTransport(
            lambda request: httpx.Response(
                503,
                json={"detail": "DeepSeek is not configured"},
                request=request,
            )
        )
    )

    with pytest.raises(PaperRAGAPIError, match="DeepSeek is not configured"):
        client.ask("question")
    http_client.close()


def test_connection_error_is_explained_without_transport_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret transport diagnostic", request=request)

    client, http_client = _client(httpx.MockTransport(handler))

    with pytest.raises(PaperRAGAPIError, match="请先启动 FastAPI") as caught:
        client.health()

    assert "secret transport diagnostic" not in str(caught.value)
    http_client.close()


@pytest.mark.parametrize("url", ["", "localhost:8000", "file:///tmp/api"])
def test_invalid_api_url_is_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="API URL"):
        PaperRAGAPIClient(url)


def test_invalid_pdf_is_rejected_before_request() -> None:
    client, http_client = _client(
        httpx.MockTransport(lambda request: httpx.Response(500, request=request))
    )

    with pytest.raises(ValueError, match="PDF"):
        client.index_pdf("notes.txt", b"hello")
    with pytest.raises(ValueError, match="empty"):
        client.index_pdf("paper.pdf", b"")
    http_client.close()
