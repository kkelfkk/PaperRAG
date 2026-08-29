"""Tests for evidence-grounded answer generation and citation validation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest

from app.generation.grounded import CitationValidationError, GroundedAnswerGenerator
from app.generation.models import GenerationConfig
from app.retrieval.models import SearchHit


class FakeLLM:
    model_name = "test/llm"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.messages: list[Sequence[Mapping[str, str]]] = []

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.messages.append(messages)
        return json.dumps(self.payload, ensure_ascii=False)


def _hit(
    rank: int = 1,
    *,
    text: str = "Retrieval augmented generation uses external evidence.",
) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=0.9,
        chunk_id=f"chunk-{rank}",
        document_id="doc-1",
        source_file="paper.pdf",
        title="RAG Paper",
        page_number=rank + 2,
        chunk_index=rank - 1,
        section="Introduction",
        text=text,
    )


def test_generate_returns_validated_citation_metadata() -> None:
    llm = FakeLLM(
        {
            "answer": "RAG uses external evidence to ground its answer. [S1]",
            "cited_source_ids": ["S1"],
            "abstained": False,
        }
    )

    result = GroundedAnswerGenerator(llm).generate("How does RAG work?", [_hit()])

    assert not result.abstained
    assert result.retrieved_count == 1
    assert result.citations[0].source_id == "S1"
    assert result.citations[0].page_number == 3
    assert result.citations[0].source_file == "paper.pdf"
    prompt = llm.messages[0][1]["content"]
    assert "[S1]" in prompt
    assert "Page: 3" in prompt
    assert "Retrieval augmented" in prompt


def test_no_hits_abstains_without_calling_llm() -> None:
    llm = FakeLLM({})

    result = GroundedAnswerGenerator(llm).generate("Unknown question", [])

    assert result.abstained
    assert not result.citations
    assert not llm.messages


@pytest.mark.parametrize(
    "payload",
    [
        {
            "answer": "Unsupported claim. [S9]",
            "cited_source_ids": ["S9"],
            "abstained": False,
        },
        {
            "answer": "Claim without a marker.",
            "cited_source_ids": ["S1"],
            "abstained": False,
        },
        {
            "answer": "Claim without evidence.",
            "cited_source_ids": [],
            "abstained": False,
        },
        {
            "answer": "Cannot answer. [S1]",
            "cited_source_ids": ["S1"],
            "abstained": True,
        },
    ],
)
def test_invalid_citations_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(CitationValidationError):
        GroundedAnswerGenerator(FakeLLM(payload)).generate("question", [_hit()])


def test_context_budget_truncates_ranked_sources() -> None:
    llm = FakeLLM(
        {
            "answer": "The first source provides evidence. [S1]",
            "cited_source_ids": ["S1"],
            "abstained": False,
        }
    )
    generator = GroundedAnswerGenerator(
        llm,
        GenerationConfig(max_context_chars=500),
    )
    hits = [_hit(1, text="evidence " * 100), _hit(2, text="agent " * 100)]

    result = generator.generate("question", hits)

    assert result.retrieved_count == 1
    prompt = llm.messages[0][1]["content"]
    assert "[S1]" in prompt
    assert "[S2]" not in prompt
    source_context = prompt.split("Sources:\n", maxsplit=1)[1]
    assert len(source_context) <= 500
