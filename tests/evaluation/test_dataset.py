"""Tests for versioned retrieval evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.evaluation.cli import main
from app.evaluation.dataset import EvaluationDataset, QuestionType, load_dataset


def _payload() -> dict[str, Any]:
    return {
        "name": "test-set",
        "version": "0.1.0",
        "description": "A manually reviewed test dataset.",
        "corpus_id": "corpus-v1",
        "queries": [
            {
                "query_id": "fact-001",
                "question": "What is the contribution?",
                "question_type": "factual",
                "relevant_chunk_ids": ["chunk-a"],
                "relevance_grades": {"chunk-a": 2},
            }
        ],
    }


def test_load_dataset_and_default_binary_grade(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    payload = _payload()
    payload["queries"][0]["relevance_grades"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = load_dataset(path)

    assert dataset.queries[0].grades() == {"chunk-a": 1}


def test_duplicate_query_ids_are_rejected() -> None:
    payload = _payload()
    payload["queries"].append(dict(payload["queries"][0]))

    with pytest.raises(ValidationError, match="query_id values must be unique"):
        EvaluationDataset.model_validate(payload)


def test_unknown_relevance_grade_is_rejected() -> None:
    payload = _payload()
    payload["queries"][0]["relevance_grades"]["unknown"] = 1

    with pytest.raises(ValidationError, match="unknown chunk IDs"):
        EvaluationDataset.model_validate(payload)


def test_invalid_dataset_version_is_rejected() -> None:
    payload = _payload()
    payload["version"] = "v1"

    with pytest.raises(ValidationError, match="version"):
        EvaluationDataset.model_validate(payload)


def test_invalid_query_page_range_is_rejected() -> None:
    payload = _payload()
    payload["queries"][0].update({"page_from": 8, "page_to": 2})

    with pytest.raises(ValidationError, match="page_from"):
        EvaluationDataset.model_validate(payload)


def test_validate_only_cli_does_not_need_retrieval_runtime(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    assert main([str(path), "--validate-only"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["query_count"] == 1


def test_committed_first_reviewed_dataset_is_valid() -> None:
    dataset = load_dataset("data/eval/paperrag_retrieval_10.json")

    assert dataset.name == "paperrag-retrieval-10"
    assert dataset.version == "0.1.0"
    assert dataset.corpus_id == "paperrag-agent-rag-v1"
    assert len(dataset.queries) == 10
    assert all(query.relevant_chunk_ids for query in dataset.queries)
    assert {query.query_id for query in dataset.queries} == {
        "rag-fact-001",
        "rag-term-002",
        "rag-fact-003",
        "rag-fact-004",
        "rag-multi-005",
        "react-term-001",
        "react-fact-002",
        "react-multi-003",
        "react-fact-004",
        "react-fact-005",
    }


def test_committed_twenty_question_dataset_extends_first_subset() -> None:
    first = load_dataset("data/eval/paperrag_retrieval_10.json")
    extended = load_dataset("data/eval/paperrag_retrieval_20.json")

    assert extended.name == "paperrag-retrieval-20"
    assert extended.version == "0.2.0"
    assert extended.corpus_id == first.corpus_id
    assert len(extended.queries) == 20
    assert [query.query_id for query in extended.queries[:10]] == [
        query.query_id for query in first.queries
    ]
    assert [query.relevant_chunk_ids for query in extended.queries[:10]] == [
        query.relevant_chunk_ids for query in first.queries
    ]


def test_committed_final_dataset_extends_twenty_question_subset() -> None:
    previous = load_dataset("data/eval/paperrag_retrieval_20.json")
    final = load_dataset("data/eval/paperrag_retrieval_30.json")

    assert final.name == "paperrag-retrieval-30"
    assert final.version == "1.0.0"
    assert final.corpus_id == previous.corpus_id
    assert len(final.queries) == 30
    assert [query.query_id for query in final.queries[:20]] == [
        query.query_id for query in previous.queries
    ]
    assert [query.relevant_chunk_ids for query in final.queries[:20]] == [
        query.relevant_chunk_ids for query in previous.queries
    ]
    assert all(
        query.question_type in {QuestionType.COMPARISON, QuestionType.MULTI_EVIDENCE}
        for query in final.queries[20:]
    )
