from __future__ import annotations

import logging

import numpy as np
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import scipy

from app import __version__
from app.config import get_settings
from app.core.diagnostics import FitError, FitStatus
from app.logging_config import configure_logging, log_event, run_id_var
from app.schemas import FitRequest, FitResponse
from app.service import fit


settings = get_settings()
configure_logging(settings.log_level, settings.log_json)
logger = logging.getLogger("nls.api")

app = FastAPI(
    title="Bounded Trust-Region Nonlinear Least Squares",
    version=__version__,
)


@app.exception_handler(FitError)
async def fit_error_handler(request: Request, exc: FitError) -> JSONResponse:
    status_code = 422
    if exc.status in {FitStatus.INVALID_INPUT, FitStatus.INVALID_EXPRESSION}:
        status_code = 400
    log_event(
        logger,
        "fit_rejected",
        level=logging.WARNING,
        path=request.url.path,
        status=exc.status.value,
        error_status=exc.status.value,
        message=exc.message,
    )
    run_id = exc.run_id or run_id_var.get()
    return JSONResponse(
        status_code=status_code,
        content={
            "run_id": run_id,
            "success": False,
            "status": exc.status.value,
            "message": exc.message,
            "reason": str(exc),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    details = exc.errors()
    log_event(
        logger,
        "request_rejected",
        level=logging.WARNING,
        path=request.url.path,
        nls_status=FitStatus.INVALID_INPUT.value,
        validation_errors=details,
    )
    return JSONResponse(
        status_code=400,
        content={
            "run_id": run_id_var.get(),
            "success": False,
            "status": FitStatus.INVALID_INPUT.value,
            "message": "request schema validation failed",
            "reason": jsonable_encoder(details),
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": __version__,
        "environment": settings.environment,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


@app.post("/api/v1/least-squares/fit", response_model=FitResponse)
def fit_least_squares(request: FitRequest) -> FitResponse:
    response, _result = fit(request)
    return response


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    run_id = run_id_var.get()
    log_event(
        logger,
        "unhandled_exception",
        level=logging.ERROR,
        path=request.url.path,
        exception_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "run_id": run_id,
            "success": False,
            "status": FitStatus.INTERNAL_ERROR.value,
            "message": "internal solver service error",
            "reason": "an unexpected error occurred; inspect service logs for run details",
        },
    )
