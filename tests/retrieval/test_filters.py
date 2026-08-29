"""Tests for shared retrieval metadata filter validation."""

from __future__ import annotations

import pytest

from app.retrieval.filters import SearchFilters


@pytest.mark.parametrize(
    "filters",
    [
        SearchFilters(document_id="doc-1"),
        SearchFilters(section="Methods"),
        SearchFilters(page_from=2),
        SearchFilters(page_to=8),
        SearchFilters(page_from=2, page_to=8),
    ],
)
def test_valid_filters_create_qdrant_conditions(filters: SearchFilters) -> None:
    query_filter = filters.to_qdrant()

    assert query_filter is not None
    assert query_filter.must


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"document_id": " "}, "document_id"),
        ({"section": " "}, "section"),
        ({"page_from": 0}, "page_from"),
        ({"page_to": -1}, "page_to"),
        ({"page_from": 9, "page_to": 3}, "greater"),
    ],
)
def test_invalid_filters_are_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SearchFilters(**kwargs)  # type: ignore[arg-type]


def test_empty_filters_do_not_create_qdrant_filter() -> None:
    assert SearchFilters().to_qdrant() is None
