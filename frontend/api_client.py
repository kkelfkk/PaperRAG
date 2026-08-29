"""Typed-enough synchronous client for PaperRAG's local HTTP API."""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"


class PaperRAGAPIError(RuntimeError):
    """A safe, user-facing API or connection error."""


class PaperRAGAPIClient:
    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        *,
        timeout: float = 120.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        normalized = base_url.strip().rstrip("/")
        parsed = httpx.URL(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("API URL must be an absolute http:// or https:// URL")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = normalized
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(
                method,
                f"{self.base_url}{path}",
                **kwargs,
            )
        except httpx.RequestError as exc:
            raise PaperRAGAPIError(
                "无法连接 PaperRAG API，请先启动 FastAPI 服务。"
            ) from exc
        if response.is_error:
            detail = ""
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    detail = str(payload.get("detail", ""))
            except ValueError:
                pass
            message = detail or f"PaperRAG API returned HTTP {response.status_code}"
            raise PaperRAGAPIError(message)
        try:
            payload = response.json()
        except ValueError as exc:
            raise PaperRAGAPIError("PaperRAG API 返回了无效的 JSON。") from exc
        if not isinstance(payload, dict):
            raise PaperRAGAPIError("PaperRAG API 返回的数据格式不正确。")
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def index_pdf(
        self,
        filename: str,
        content: bytes,
        *,
        recreate: bool = False,
        max_chunk_chars: int = 1200,
        overlap_chars: int = 200,
    ) -> dict[str, Any]:
        if not filename.casefold().endswith(".pdf"):
            raise ValueError("Only PDF files can be indexed")
        if not content:
            raise ValueError("PDF content cannot be empty")
        return self._request(
            "POST",
            "/v1/documents/index",
            files={"file": (filename, content, "application/pdf")},
            data={
                "recreate": str(recreate).lower(),
                "max_chunk_chars": str(max_chunk_chars),
                "overlap_chars": str(overlap_chars),
            },
        )

    @staticmethod
    def _query_payload(
        query: str,
        *,
        top_k: int,
        strategy: str,
        document_id: str | None,
        section: str | None,
        page_from: int | None,
        page_to: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "strategy": strategy,
        }
        optional = {
            "document_id": document_id,
            "section": section,
            "page_from": page_from,
            "page_to": page_to,
        }
        payload.update(
            {
                key: value
                for key, value in optional.items()
                if value is not None and value != ""
            }
        )
        return payload

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        strategy: str = "hybrid",
        document_id: str | None = None,
        section: str | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/search",
            json=self._query_payload(
                query,
                top_k=top_k,
                strategy=strategy,
                document_id=document_id,
                section=section,
                page_from=page_from,
                page_to=page_to,
            ),
        )

    def ask(
        self,
        query: str,
        *,
        top_k: int = 5,
        strategy: str = "hybrid",
        document_id: str | None = None,
        section: str | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/ask",
            json=self._query_payload(
                query,
                top_k=top_k,
                strategy=strategy,
                document_id=document_id,
                section=section,
                page_from=page_from,
                page_to=page_to,
            ),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
