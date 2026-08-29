"""HTTP contract tests for the DeepSeek client without real API calls."""

from __future__ import annotations

import json

import httpx
import pytest

from app.generation.llm import DeepSeekClient, LLMError


def test_deepseek_client_sends_secret_in_header_and_requests_json() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"answer":"ok","cited_source_ids":[],"abstained":true}'}}
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = DeepSeekClient(
        "test-secret-key",
        model_name="deepseek-v4-flash",
        http_client=http_client,
    )

    content = client.complete([{"role": "user", "content": "hello"}])

    assert json.loads(content)["answer"] == "ok"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["authorization"] == "Bearer test-secret-key"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "deepseek-v4-flash"
    assert body["thinking"] == {"type": "disabled"}
    assert body["response_format"] == {"type": "json_object"}
    assert body["stream"] is False
    http_client.close()


def test_deepseek_error_omits_response_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(401, text="secret diagnostic response")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = DeepSeekClient("fake-key", http_client=http_client)

    with pytest.raises(LLMError, match="HTTP 401") as caught:
        client.complete([{"role": "user", "content": "hello"}])

    assert "secret diagnostic response" not in str(caught.value)
    http_client.close()
