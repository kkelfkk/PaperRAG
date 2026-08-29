"""FastAPI application for PaperRAG's local baseline."""

from __future__ import annotations

import tempfile
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import (
    AnswerResponse,
    AskRequest,
    HealthResponse,
    IndexResponse,
    SearchRequest,
    SearchResponse,
)
from app.api.service import PaperRAGService, create_default_service
from app.generation.llm import LLMError
from app.ingestion.pdf_parser import PDFParseError
from app.retrieval.dense import DenseRetrievalError

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
UPLOAD_BLOCK_BYTES = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.service = None
    app.state.service_lock = threading.Lock()
    yield
    service = app.state.service
    if service is not None:
        service.close()


def get_service(request: Request) -> PaperRAGService:
    """Lazily load the model and database only when a pipeline route needs it."""

    service = request.app.state.service
    if service is not None:
        return service
    with request.app.state.service_lock:
        service = request.app.state.service
        if service is None:
            service = create_default_service()
            request.app.state.service = service
    return service


def _raise_service_error(exc: Exception) -> None:
    if isinstance(exc, LLMError):
        status = 503 if "not configured" in str(exc) else 502
    elif isinstance(exc, PDFParseError):
        status = 422
    else:
        status = 400
    raise HTTPException(status_code=status, detail=str(exc)) from exc


def create_app() -> FastAPI:
    application = FastAPI(
        title="PaperRAG API",
        version="0.1.0",
        description="Evidence-grounded paper retrieval and question answering.",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service="paperrag", version="0.1.0")

    @application.post(
        "/v1/documents/index",
        response_model=IndexResponse,
        tags=["documents"],
    )
    async def index_document(
        file: Annotated[UploadFile, File(description="A text-based academic PDF")],
        service: Annotated[PaperRAGService, Depends(get_service)],
        recreate: Annotated[bool, Form()] = False,
        max_chunk_chars: Annotated[int, Form(ge=64, le=20000)] = 1200,
        overlap_chars: Annotated[int, Form(ge=0, le=10000)] = 200,
    ) -> IndexResponse:
        filename = file.filename or "uploaded.pdf"
        if Path(filename).suffix.casefold() != ".pdf":
            await file.close()
            raise HTTPException(status_code=415, detail="Only .pdf uploads are accepted")

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                received = 0
                while block := await file.read(UPLOAD_BLOCK_BYTES):
                    received += len(block)
                    if received > MAX_UPLOAD_BYTES:
                        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
                        raise HTTPException(
                            status_code=413,
                            detail=f"PDF exceeds the {limit_mb} MB limit",
                        )
                    temp_file.write(block)
            result = await run_in_threadpool(
                service.index_pdf,
                temp_path,
                source_name=filename,
                max_chunk_chars=max_chunk_chars,
                overlap_chars=overlap_chars,
                recreate=recreate,
            )
            return IndexResponse.model_validate(result.to_dict())
        except HTTPException:
            raise
        except (DenseRetrievalError, PDFParseError, OSError, ValueError) as exc:
            _raise_service_error(exc)
        finally:
            await file.close()
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @application.post("/v1/search", response_model=SearchResponse, tags=["retrieval"])
    def search(
        request: SearchRequest,
        service: Annotated[PaperRAGService, Depends(get_service)],
    ) -> SearchResponse:
        try:
            result = service.search(
                request.query,
                top_k=request.top_k,
                document_id=request.document_id,
                strategy=request.strategy,
            )
            return SearchResponse.model_validate(result.to_dict())
        except (DenseRetrievalError, OSError, ValueError) as exc:
            _raise_service_error(exc)

    @application.post("/v1/ask", response_model=AnswerResponse, tags=["generation"])
    def ask(
        request: AskRequest,
        service: Annotated[PaperRAGService, Depends(get_service)],
    ) -> AnswerResponse:
        try:
            result = service.ask(
                request.query,
                top_k=request.top_k,
                document_id=request.document_id,
                strategy=request.strategy,
            )
            return AnswerResponse.model_validate(result.to_dict())
        except (DenseRetrievalError, LLMError, OSError, ValueError) as exc:
            _raise_service_error(exc)

    return application


app = create_app()
