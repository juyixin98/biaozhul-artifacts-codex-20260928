"""FastAPI entry point."""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import get_settings
from .models import HealthResponse, SolveRequest, SolveResponse
from .solver import solve_lp

settings = get_settings()
logger = logging.getLogger("lp_service")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Small Two-Phase Simplex LP Service",
        version="1.0.0",
        description="Bounded linear programming with independent evidence checks.",
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", "unknown")
        # Do not log request values: they may include sensitive coefficients.
        logger.warning(
            "invalid_request request_id=%s error_count=%s",
            request_id,
            len(exc.errors()),
        )
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id,
                "status": "invalid",
                "message": "request validation failed",
                "errors": exc.errors(),
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service=settings.service_name)

    @app.post("/solve", response_model=SolveResponse)
    async def solve(request: SolveRequest, http_request: Request) -> SolveResponse:
        request_id = getattr(http_request.state, "request_id", "unknown")
        if request.request_id is None:
            request.request_id = request_id
        else:
            request_id = request.request_id
        try:
            response = solve_lp(request)
        except ValueError as exc:
            logger.warning("invalid_model request_id=%s", request_id)
            return JSONResponse(
                status_code=400,
                content={
                    "request_id": request_id,
                    "status": "invalid",
                    "message": str(exc),
                },
            )
        logger.info(
            "solved request_id=%s status=%s iterations=%s pivots=%s redundant_rows=%s",
            request_id,
            response.status,
            response.iterations,
            response.pivots,
            len(response.redundant_rows),
        )
        return response

    return app


app = create_app()
