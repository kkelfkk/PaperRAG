"""LLM provider interfaces and the DeepSeek Chat Completions client."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import httpx

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


class LLMError(RuntimeError):
    """Raised when an LLM request or response is invalid."""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal provider interface required by grounded generation."""

    @property
    def model_name(self) -> str:
        """Return the model identifier used for generation."""

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return one non-streaming JSON-formatted completion."""


class DeepSeekClient:
    """Small, secret-safe client for DeepSeek's OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        model_name: str = DEFAULT_DEEPSEEK_MODEL,
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        timeout_seconds: float = 60.0,
        max_tokens: int = 800,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMError("DeepSeek API key cannot be empty")
        if not model_name.strip():
            raise LLMError("DeepSeek model name cannot be empty")
        if max_tokens <= 0:
            raise LLMError("max_tokens must be positive")
        self._api_key = api_key
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout_seconds)

    @classmethod
    def from_env(
        cls,
        *,
        model_name: str | None = None,
        base_url: str | None = None,
    ) -> DeepSeekClient:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise LLMError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add "
                "a newly created key. Never commit .env."
            )
        return cls(
            api_key,
            model_name=model_name or os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            base_url=base_url
            or os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        if not messages:
            raise LLMError("messages cannot be empty")
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": [dict(message) for message in messages],
            "thinking": {"type": "disabled"},
            "temperature": 0.0,
            "max_tokens": self._max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"DeepSeek request failed: {type(exc).__name__}") from exc

        if response.status_code >= 400:
            raise LLMError(
                f"DeepSeek API returned HTTP {response.status_code}; response body "
                "was omitted to avoid leaking request details"
            )
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("DeepSeek returned an invalid chat completion response") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("DeepSeek returned empty completion content")
        return content

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
