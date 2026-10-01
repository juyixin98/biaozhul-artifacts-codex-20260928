"""FastAPI application entry point."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import ALGORITHM_VERSION, SCHEMA_VERSION, SERVICE_NAME, get_settings
from .contracts import AnalysisRequest, AnalysisResponse, HealthResponse
from .errors import AnalysisError
from .repository import AnalysisRepository
from .service import result_to_response, run_analysis


logger = logging.getLogger("lag_correlation")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.repository = AnalysisRepository(settings.database_path)
    logger.info(
        "service ready database_path=%s schema=%s algorithm=%s",
        _redact_path(settings.database_path),
        SCHEMA_VERSION,
        ALGORITHM_VERSION,
    )
    yield


app = FastAPI(
    title="Irregular Sample Lag Correlation",
    version="1.0.0",
    lifespan=lifespan,
)


def get_repository(request: Request) -> AnalysisRepository:
    return request.app.state.repository


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        service=SERVICE_NAME,
        status="ok",
        schema_version=SCHEMA_VERSION,
        algorithm_version=ALGORITHM_VERSION,
    )


@app.post(
    "/api/v1/lag-correlation/analyses",
    response_model=AnalysisResponse,
    status_code=200,
)
def create_analysis(
    request: AnalysisRequest,
    repository: AnalysisRepository = Depends(get_repository),
) -> dict:
    request_id = request.request_id or str(uuid.uuid4())
    result = run_analysis(request.model_copy(update={"request_id": request_id}))
    repository.save(result)
    logger.info(
        "analysis accepted request_id=%s status=%s anchor=%s valid_lags=%s evaluated_lags=%s",
        result.request_id,
        result.status,
        result.anchor_side,
        result.valid_lag_count,
        result.evaluated_lag_count,
    )
    return result_to_response(result)


@app.get(
    "/api/v1/lag-correlation/analyses/{request_id}",
    response_model=AnalysisResponse,
)
def get_analysis(request_id: str, repository: AnalysisRepository = Depends(get_repository)) -> dict:
    return repository.get_response_dict(request_id)


@app.exception_handler(AnalysisError)
async def analysis_error_handler(_: Request, exc: AnalysisError) -> JSONResponse:
    logger.info("analysis rejected error=%s message=%s", exc.code, str(exc))
    return JSONResponse(
        status_code=exc.http_status,
        content={
            "status": "error",
            "failure_category": exc.code,
            "message": str(exc),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    redacted_errors = [_redact_validation_error(error) for error in exc.errors()]
    logger.info("request rejected error=VALIDATION_ERROR details=%s", redacted_errors)
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "failure_category": "VALIDATION_ERROR",
            "errors": redacted_errors,
        },
    )


def _redact_validation_error(error: dict) -> dict:
    return {
        "type": error.get("type"),
        "loc": [str(item) for item in error.get("loc", [])],
        "msg": error.get("msg"),
    }


def _redact_path(path: str) -> str:
    if not path:
        return "<unset>"
    directory, separator, filename = path.rpartition("/")
    shown_directory = directory[-12:] if len(directory) > 12 else directory
    return f"{shown_directory}{separator}{filename}"
