"""Versioned, validated retrieval evaluation dataset format."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class QuestionType(StrEnum):
    FACTUAL = "factual"
    TERMINOLOGY = "terminology"
    COMPARISON = "comparison"
    MULTI_EVIDENCE = "multi_evidence"


class EvaluationQuery(BaseModel):
    """One question with manually verified relevant chunk IDs."""

    query_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)
    question_type: QuestionType
    relevant_chunk_ids: list[str] = Field(min_length=1)
    relevance_grades: dict[str, int] = Field(default_factory=dict)
    document_id: str | None = Field(default=None, max_length=128)
    notes: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_relevance(self) -> EvaluationQuery:
        if len(self.relevant_chunk_ids) != len(set(self.relevant_chunk_ids)):
            raise ValueError("relevant_chunk_ids cannot contain duplicates")
        unknown = set(self.relevance_grades) - set(self.relevant_chunk_ids)
        if unknown:
            raise ValueError(
                f"relevance_grades contains unknown chunk IDs: {sorted(unknown)}"
            )
        if any(grade <= 0 for grade in self.relevance_grades.values()):
            raise ValueError("relevance grades must be positive integers")
        return self

    def grades(self) -> dict[str, int]:
        """Return explicit grades with binary relevance as the default."""

        return {
            chunk_id: self.relevance_grades.get(chunk_id, 1)
            for chunk_id in self.relevant_chunk_ids
        }


class EvaluationDataset(BaseModel):
    """A reproducible set of labeled retrieval questions."""

    name: str = Field(min_length=1, max_length=200)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str = Field(min_length=1, max_length=2000)
    corpus_id: str = Field(min_length=1, max_length=200)
    queries: list[EvaluationQuery] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_query_ids(self) -> EvaluationDataset:
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_id values must be unique within a dataset")
        return self


def load_dataset(path: str | Path) -> EvaluationDataset:
    """Load and validate a UTF-8 JSON evaluation dataset."""

    dataset_path = Path(path).expanduser()
    if not dataset_path.exists():
        raise ValueError(f"evaluation dataset does not exist: {dataset_path}")
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"evaluation dataset is not valid JSON: {exc}") from exc
    return EvaluationDataset.model_validate(payload)
