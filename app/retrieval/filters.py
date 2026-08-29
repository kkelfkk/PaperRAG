"""Shared, validated metadata filters for every retrieval strategy."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import models


@dataclass(frozen=True, slots=True)
class SearchFilters:
    document_id: str | None = None
    section: str | None = None
    page_from: int | None = None
    page_to: int | None = None

    def __post_init__(self) -> None:
        if self.document_id is not None and not self.document_id.strip():
            raise ValueError("document_id filter cannot be empty")
        if self.section is not None and not self.section.strip():
            raise ValueError("section filter cannot be empty")
        if self.page_from is not None and self.page_from <= 0:
            raise ValueError("page_from must be positive")
        if self.page_to is not None and self.page_to <= 0:
            raise ValueError("page_to must be positive")
        if (
            self.page_from is not None
            and self.page_to is not None
            and self.page_from > self.page_to
        ):
            raise ValueError("page_from cannot be greater than page_to")

    def to_qdrant(self) -> models.Filter | None:
        conditions: list[models.Condition] = []
        if self.document_id:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=self.document_id),
                )
            )
        if self.section:
            conditions.append(
                models.FieldCondition(
                    key="section",
                    match=models.MatchValue(value=self.section),
                )
            )
        if self.page_from is not None or self.page_to is not None:
            conditions.append(
                models.FieldCondition(
                    key="page_number",
                    range=models.Range(gte=self.page_from, lte=self.page_to),
                )
            )
        return models.Filter(must=conditions) if conditions else None
