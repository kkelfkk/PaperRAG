"""Tests for versioned retrieval evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.evaluation.cli import main
from app.evaluation.dataset import EvaluationDataset, load_dataset


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
