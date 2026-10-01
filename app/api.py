"""HTTP API for probability calibration runs."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import settings
from .contracts import ErrorBody, RunRequest, RunResult, RunSummary
from .errors import CalibrationError
from .logging_setup import configure_logging
from .repository import RunRepository
from .service import CalibrationService

configure_logging(level=settings.log_level, log_file=settings.log_file)
logger = logging.getLogger("calibration.api")
repository = RunRepository(settings.database_path)
repository.initialize()
service = CalibrationService(settings, repository)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Weighted fixed-bin and equal-frequency calibration diagnostics.",
)


@app.exception_handler(CalibrationError)
async def calibration_error_handler(_: Request, exc: CalibrationError) -> JSONResponse:
    logger.warning(
        "request failed with a domain error",
        extra={"extra_fields": {"category": exc.category, "code": exc.code}},
    )
    body = ErrorBody(
        category=exc.category,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("request failed schema validation")
    body = ErrorBody(
        category="VALIDATION_ERROR",
        code="request_schema_violation",
        message="request body did not satisfy the sample contract",
        details={"errors": exc.errors()},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=body.model_dump(mode="json"),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs", response_model=RunResult, status_code=status.HTTP_201_CREATED)
def create_run(request: RunRequest) -> RunResult:
    return service.create_run(request)


@app.get("/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    return service.list_runs()


@app.get("/runs/{run_id}", response_model=RunResult)
def get_run(run_id: str) -> RunResult:
    return service.get_run(run_id)
