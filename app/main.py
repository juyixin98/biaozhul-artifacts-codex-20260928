"""FastAPI entry point."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app import __version__
from app.observability import RunLogger
from app.schemas import SolveRequest
from app.service import solve_request
from app.status import FAILING_STATUSES, SolverError, Status

logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(
    title="Bounded Nonlinear Least Squares Trust-Region Service",
    version=__version__,
)

_STATUS_CODES = {
    Status.INVALID_REQUEST: 400,
    Status.INVALID_EXPRESSION: 400,
    Status.NONCONVERGENT: 422,
    Status.UNIDENTIFIABLE: 422,
    Status.ILLEGAL_FUNCTION_VALUE: 422,
    Status.DERIVATIVE_CHECK_FAILED: 422,
    Status.INTERNAL_ERROR: 500,
}


def _run_id(request: Request) -> str:
    return request.headers.get("X-Run-ID") or f"local-{uuid.uuid4()}"


def _error_body(run_id: str, status: Status, message: str, details: dict[str, Any], logger: RunLogger | None):
    return {
        "run_id": run_id,
        "status": status.value,
        "message": message,
        "details": details,
        "log_tail": logger.events[-20:] if logger else [],
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    run_id = _run_id(request)
    logger = RunLogger(run_id)
    logger.event("invalid_request", errors=exc.errors(), decision="schema rejected before solving")
    return JSONResponse(
        status_code=422,
        content=_error_body(
            run_id,
            Status.INVALID_REQUEST,
            "request schema validation failed",
            {"validation_errors": exc.errors()},
            logger,
        ),
    )


@app.exception_handler(SolverError)
async def solver_error_handler(request: Request, exc: SolverError):
    run_id = getattr(request.state, "run_id", _run_id(request))
    logger = getattr(request.state, "logger", None)
    if logger:
        logger.event(
            "solver_error",
            status=exc.status.value,
            message=exc.message,
            details=exc.details,
            decision="returned explicit failure category",
        )
    return JSONResponse(
        status_code=exc.http_status_code,
        content=_error_body(run_id, exc.status, exc.message, exc.details, logger),
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    run_id = _run_id(request)
    logger = RunLogger(run_id)
    logger.event("internal_error", error_type=type(exc).__name__, message=str(exc), decision="not reported as success")
    return JSONResponse(
        status_code=500,
        content=_error_body(
            run_id,
            Status.INTERNAL_ERROR,
            "unexpected internal error",
            {"error_type": type(exc).__name__},
            logger,
        ),
    )


@app.middleware("http")
async def attach_run(request: Request, call_next):
    request.state.run_id = _run_id(request)
    request.state.logger = RunLogger(request.state.run_id)
    request.state.logger.event(
        "request_received",
        method=request.method,
        path=request.url.path,
        service_version=__version__,
    )
    return await call_next(request)


@app.get("/health")
async def health(request: Request):
    return {"status": "ok", "run_id": request.state.run_id, "service_version": __version__}


@app.post("/solve")
async def solve(request: Request, payload: dict):
    try:
        model_request = SolveRequest.model_validate(payload)
    except ValidationError as exc:
        raise SolverError(
            Status.INVALID_REQUEST,
            "request schema validation failed",
            details={"validation_errors": exc.errors()},
            http_status_code=422,
        ) from exc

    result = solve_request(model_request, request.state.run_id, request.state.logger)
    status_code = 422 if Status(result["status"]) in FAILING_STATUSES else 200
    return JSONResponse(status_code=status_code, content=result)
