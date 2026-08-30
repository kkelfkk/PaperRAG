"""Tests for deterministic generation and citation evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.generation import (
    GenerationEvaluationDataset,
    GenerationPredictionSet,
    evaluate_generation,
    load_generation_dataset,
    load_generation_predictions,
)


def _dataset() -> GenerationEvaluationDataset:
    return GenerationEvaluationDataset.model_validate(
        {
            "name": "generation-dev",
            "version": "0.1.0",
            "description": "test labels",
            "corpus_id": "corpus-v1",
            "queries": [
                {
                    "query_id": "answerable",
                    "question": "What is the method?",
                    "expected_abstention": False,
                    "relevant_chunk_ids": ["gold-1", "gold-2"],
                },
                {
                    "query_id": "unknown",
                    "question": "What is not covered?",
                    "expected_abstention": True,
                    "relevant_chunk_ids": [],
                },
            ],
        }
    )


def _predictions() -> GenerationPredictionSet:
    return GenerationPredictionSet.model_validate(
        {
            "dataset_name": "generation-dev",
            "dataset_version": "0.1.0",
            "system_name": "hybrid-rerank-v1",
            "model": "test-llm",
            "predictions": [
                {
                    "query_id": "answerable",
                    "answer": "An answer [S1] [S2].",
                    "abstained": False,
                    "cited_chunk_ids": ["gold-1", "wrong"],
                    "retrieved_chunk_ids": ["gold-1", "wrong", "candidate"],
                    "evidence": {"gold-1": "support", "wrong": "not support"},
                },
                {
                    "query_id": "unknown",
                    "answer": "Insufficient evidence.",
                    "abstained": True,
                    "cited_chunk_ids": [],
                    "retrieved_chunk_ids": [],
                },
            ],
        }
    )


def test_generation_metrics_score_citations_and_abstention() -> None:
    report = evaluate_generation(_dataset(), _predictions())

    assert report.query_count == 2
    assert report.summary == {
        "abstention_correct": 1.0,
        "citation_precision": 0.5,
        "citation_recall": 0.5,
        "citation_f1": 0.5,
        "citation_validity": 1.0,
    }
    assert report.queries[1].metrics["citation_precision"] is None


def test_citation_validity_detects_sources_outside_retrieval() -> None:
    payload = _predictions().model_dump()
    payload["predictions"][0]["cited_chunk_ids"] = ["gold-1", "invented"]

    report = evaluate_generation(
        _dataset(), GenerationPredictionSet.model_validate(payload)
    )

    assert report.queries[0].metrics["citation_validity"] == 0.5


def test_answerable_query_requires_relevance_labels() -> None:
    payload = _dataset().model_dump()
    payload["queries"][0]["relevant_chunk_ids"] = []

    with pytest.raises(ValidationError, match="answerable queries"):
        GenerationEvaluationDataset.model_validate(payload)


def test_prediction_evidence_must_have_been_retrieved() -> None:
    payload = _predictions().model_dump()
    payload["predictions"][0]["evidence"]["invented"] = "text"

    with pytest.raises(ValidationError, match="absent from retrieved"):
        GenerationPredictionSet.model_validate(payload)


def test_evaluation_rejects_missing_prediction() -> None:
    payload = _predictions().model_dump()
    payload["predictions"].pop()

    with pytest.raises(ValueError, match="missing=.*unknown"):
        evaluate_generation(
            _dataset(), GenerationPredictionSet.model_validate(payload)
        )


def test_generation_files_load_and_cli_writes_report(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "labels.json"
    predictions_path = tmp_path / "predictions.json"
    output_path = tmp_path / "report.json"
    dataset_path.write_text(_dataset().model_dump_json(), encoding="utf-8")
    predictions_path.write_text(_predictions().model_dump_json(), encoding="utf-8")

    assert load_generation_dataset(dataset_path).name == "generation-dev"
    assert load_generation_predictions(predictions_path).model == "test-llm"

    from app.evaluation.generation_cli import main

    assert main([str(dataset_path), str(predictions_path), "-o", str(output_path)]) == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["query_count"] == 2


def test_generation_run_cli_rejects_missing_collection(tmp_path: Path) -> None:
    dataset_path = tmp_path / "labels.json"
    dataset_path.write_text(_dataset().model_dump_json(), encoding="utf-8")

    from app.evaluation.generation_run_cli import main

    assert (
        main(
            [
                str(dataset_path),
                "--output",
                str(tmp_path / "predictions.json"),
                "--db-path",
                str(tmp_path / "qdrant"),
                "--collection",
                "missing",
            ]
        )
        == 2
    )
    assert not (tmp_path / "predictions.json").exists()
