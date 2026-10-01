"""FastAPI composition root and HTTP boundary."""

from __future__ import annotations

import math
import os
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.errors import DomainError, InvalidInputError
from app.schemas import (
    BatchRequest,
    BatchResponse,
    CreateRunRequest,
    ErrorResponse,
    PosteriorSummaryResponse,
    PredictionRequest,
    PredictionResponse,
    RunResponse,
    RunVersionResponse,
)
from app.service import BetaBinomialService
from app.storage import SQLiteStore


def default_database_path() -> Path:
    return Path(
        os.environ.get("BETA_BINOMIAL_DB", "data/beta_binomial.sqlite3")
    )


def create_app(service: BetaBinomialService | None = None) -> FastAPI:
    app = FastAPI(
        title="Beta-Binomial Batch Update API",
        version="0.1.0",
    )
    app.state.service = service

    def get_service() -> BetaBinomialService:
        if app.state.service is None:
            app.state.service = BetaBinomialService.with_file_database(
                default_database_path()
            )
        return app.state.service

    def _json_safe(value):
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        if isinstance(value, Exception):
            return f"{type(value).__name__}: {value}"
        if isinstance(value, float) and not math.isfinite(value):
            return "Infinity" if value > 0 else "-Infinity"
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    @app.exception_handler(DomainError)
    async def domain_error_handler(
        request: Request, exc: DomainError
    ) -> JSONResponse:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        payload = ErrorResponse(
            error=exc.message,
            category=exc.category,
            details=exc.details,
            request_id=request_id,
        )
        return JSONResponse(
            status_code=exc.status_code, content=payload.model_dump()
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        details = {"validation_errors": _json_safe(exc.errors())}
        payload = ErrorResponse(
            error="Request does not satisfy the sample or parameter contract.",
            category=InvalidInputError.category,
            details=details,
            request_id=request_id,
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.post("/runs", response_model=RunResponse, tags=["runs"])
    def create_run(
        request: CreateRunRequest,
        service: BetaBinomialService = Depends(get_service),
    ) -> RunResponse:
        return service.create_run(request.run_id, request.alpha, request.beta)

    @app.get("/runs/{run_id}", response_model=RunResponse, tags=["runs"])
    def get_run(
        run_id: str, service: BetaBinomialService = Depends(get_service)
    ) -> RunResponse:
        return service.get_run(run_id)

    @app.post(
        "/runs/{run_id}/batches/{batch_id}",
        response_model=BatchResponse,
        tags=["batches"],
    )
    def submit_batch(
        run_id: str,
        batch_id: str,
        request: BatchRequest,
        service: BetaBinomialService = Depends(get_service),
    ) -> BatchResponse:
        return service.submit_batch(
            run_id,
            batch_id,
            request.successes,
            request.failures,
            request.reason,
        )

    @app.put(
        "/runs/{run_id}/batches/{batch_id}",
        response_model=BatchResponse,
        tags=["batches"],
    )
    def revise_batch(
        run_id: str,
        batch_id: str,
        request: BatchRequest,
        service: BetaBinomialService = Depends(get_service),
    ) -> BatchResponse:
        return service.revise_batch(
            run_id,
            batch_id,
            request.successes,
            request.failures,
            request.reason,
        )

    @app.get(
        "/runs/{run_id}/posterior",
        response_model=PosteriorSummaryResponse,
        tags=["inference"],
    )
    def posterior(
        run_id: str,
        confidence_level: float = 0.95,
        service: BetaBinomialService = Depends(get_service),
    ) -> PosteriorSummaryResponse:
        return service.posterior_summary(run_id, confidence_level)

    @app.post(
        "/runs/{run_id}/predictions",
        response_model=PredictionResponse,
        tags=["inference"],
    )
    def predict(
        run_id: str,
        request: PredictionRequest,
        service: BetaBinomialService = Depends(get_service),
    ) -> PredictionResponse:
        return service.predict(
            run_id, request.future_n, request.confidence_level
        )

    @app.get(
        "/runs/{run_id}/versions",
        response_model=list[RunVersionResponse],
        tags=["versions"],
    )
    def versions(
        run_id: str, service: BetaBinomialService = Depends(get_service)
    ) -> list[RunVersionResponse]:
        return service.list_versions(run_id)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
