"""Validated HTTP request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.retrieval.filters import SearchFilters


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class IndexResponse(BaseModel):
    collection_name: str
    document_id: str
    indexed_chunks: int
    vector_size: int
    embedding_model: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: str | None = Field(default=None, max_length=128)
    section: str | None = Field(default=None, min_length=1, max_length=500)
    page_from: int | None = Field(default=None, ge=1)
    page_to: int | None = Field(default=None, ge=1)
    strategy: Literal[
        "dense",
        "bm25",
        "hybrid",
        "hybrid_rerank",
        "decomposed_hybrid_rerank",
    ] = "hybrid"

    @model_validator(mode="after")
    def validate_page_range(self) -> SearchRequest:
        if (
            self.page_from is not None
            and self.page_to is not None
            and self.page_from > self.page_to
        ):
            raise ValueError("page_from cannot be greater than page_to")
        return self

    def to_search_filters(self) -> SearchFilters:
        return SearchFilters(
            document_id=self.document_id,
            section=self.section,
            page_from=self.page_from,
            page_to=self.page_to,
        )


class SearchHitResponse(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_id: str
    source_file: str
    title: str
    page_number: int
    chunk_index: int
    section: str | None
    text: str


class SearchResponse(BaseModel):
    query: str
    collection_name: str
    embedding_model: str
    hits: list[SearchHitResponse]


class AskRequest(SearchRequest):
    pass


class CitationResponse(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    source_file: str
    title: str
    page_number: int
    section: str | None


class AnswerResponse(BaseModel):
    query: str
    answer: str
    abstained: bool
    model: str
    retrieved_count: int
    citations: list[CitationResponse]
