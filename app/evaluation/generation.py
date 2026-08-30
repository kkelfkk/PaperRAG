"""Validated offline evaluation for generated answers and citations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class GenerationEvaluationQuery(BaseModel):
    """Human-reviewed expectation for one generation question."""

    query_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)
    expected_abstention: bool = False
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    reference_answer: str | None = Field(default=None, min_length=1, max_length=8000)
    notes: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_labels(self) -> GenerationEvaluationQuery:
        if len(self.relevant_chunk_ids) != len(set(self.relevant_chunk_ids)):
            raise ValueError("relevant_chunk_ids cannot contain duplicates")
        if not self.expected_abstention and not self.relevant_chunk_ids:
            raise ValueError("answerable queries require relevant_chunk_ids")
        return self


class GenerationEvaluationDataset(BaseModel):
    """Versioned human labels for answer and citation evaluation."""

    name: str = Field(min_length=1, max_length=200)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str = Field(min_length=1, max_length=2000)
    corpus_id: str = Field(min_length=1, max_length=200)
    queries: list[GenerationEvaluationQuery] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_query_ids(self) -> GenerationEvaluationDataset:
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_id values must be unique within a dataset")
        return self


class GenerationPrediction(BaseModel):
    """One saved system output; evidence text enables later LLM judging."""

    query_id: str = Field(min_length=1, max_length=128)
    answer: str = Field(min_length=1, max_length=20000)
    abstained: bool
    cited_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prediction(self) -> GenerationPrediction:
        for field_name, values in (
            ("cited_chunk_ids", self.cited_chunk_ids),
            ("retrieved_chunk_ids", self.retrieved_chunk_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} cannot contain duplicates")
        unknown_evidence = set(self.evidence) - set(self.retrieved_chunk_ids)
        if unknown_evidence:
            raise ValueError(
                "evidence contains chunks absent from retrieved_chunk_ids: "
                f"{sorted(unknown_evidence)}"
            )
        if any(not text.strip() for text in self.evidence.values()):
            raise ValueError("evidence text cannot be empty")
        return self


class GenerationPredictionSet(BaseModel):
    """Saved outputs from one fixed PaperRAG configuration."""

    dataset_name: str = Field(min_length=1, max_length=200)
    dataset_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    system_name: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=500)
    predictions: list[GenerationPrediction] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_query_ids(self) -> GenerationPredictionSet:
        query_ids = [prediction.query_id for prediction in self.predictions]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_id values must be unique within predictions")
        return self


@dataclass(frozen=True, slots=True)
class GenerationQueryEvaluation:
    query_id: str
    question: str
    expected_abstention: bool
    predicted_abstention: bool
    relevant_chunk_ids: tuple[str, ...]
    cited_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    metrics: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class GenerationEvaluationReport:
    dataset_name: str
    dataset_version: str
    corpus_id: str
    system_name: str
    model: str
    query_count: int
    summary: dict[str, float]
    queries: tuple[GenerationQueryEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: str | Path, description: str) -> Any:
    input_path = Path(path).expanduser()
    if not input_path.exists():
        raise ValueError(f"{description} does not exist: {input_path}")
    try:
        return json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {exc}") from exc


def load_generation_dataset(path: str | Path) -> GenerationEvaluationDataset:
    return GenerationEvaluationDataset.model_validate(
        _load_json(path, "generation evaluation dataset")
    )


def load_generation_predictions(path: str | Path) -> GenerationPredictionSet:
    return GenerationPredictionSet.model_validate(
        _load_json(path, "generation predictions")
    )


def _citation_metrics(
    query: GenerationEvaluationQuery,
    prediction: GenerationPrediction,
) -> dict[str, float | None]:
    cited = set(prediction.cited_chunk_ids)
    relevant = set(query.relevant_chunk_ids)
    retrieved = set(prediction.retrieved_chunk_ids)
    correct = len(cited & relevant)

    validity = (
        len(cited & retrieved) / len(cited)
        if cited
        else float(prediction.abstained)
    )
    if query.expected_abstention:
        precision = recall = f1 = None
    else:
        precision = correct / len(cited) if cited else 0.0
        recall = correct / len(relevant)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "abstention_correct": float(
            prediction.abstained == query.expected_abstention
        ),
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
        "citation_validity": validity,
    }


def evaluate_generation(
    dataset: GenerationEvaluationDataset,
    predictions: GenerationPredictionSet,
) -> GenerationEvaluationReport:
    """Score saved outputs without making another model or network call."""

    if predictions.dataset_name != dataset.name:
        raise ValueError("prediction dataset_name does not match labels")
    if predictions.dataset_version != dataset.version:
        raise ValueError("prediction dataset_version does not match labels")

    prediction_map = {
        prediction.query_id: prediction for prediction in predictions.predictions
    }
    expected_ids = {query.query_id for query in dataset.queries}
    actual_ids = set(prediction_map)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"prediction query IDs do not match labels; missing={missing}, "
            f"unexpected={unexpected}"
        )

    results = tuple(
        GenerationQueryEvaluation(
            query_id=query.query_id,
            question=query.question,
            expected_abstention=query.expected_abstention,
            predicted_abstention=prediction_map[query.query_id].abstained,
            relevant_chunk_ids=tuple(query.relevant_chunk_ids),
            cited_chunk_ids=tuple(prediction_map[query.query_id].cited_chunk_ids),
            retrieved_chunk_ids=tuple(
                prediction_map[query.query_id].retrieved_chunk_ids
            ),
            metrics=_citation_metrics(query, prediction_map[query.query_id]),
        )
        for query in dataset.queries
    )
    metric_names = tuple(results[0].metrics)
    summary = {
        name: sum(values) / len(values)
        for name in metric_names
        if (
            values := [
                value
                for result in results
                if (value := result.metrics[name]) is not None
            ]
        )
    }
    return GenerationEvaluationReport(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        corpus_id=dataset.corpus_id,
        system_name=predictions.system_name,
        model=predictions.model,
        query_count=len(results),
        summary=summary,
        queries=results,
    )
