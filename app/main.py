"""FastAPI application entrypoint."""

from __future__ import annotations

import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import Settings
from .contracts import (
    ErrorResponse,
    SurvivalAnalysisRequest,
    SurvivalAnalysisResponse,
    SyntheticDatasetRequest,
)
from .logging_config import configure_logging
from .repository import AnalysisNotFound, SurvivalRepository
from .service import run_analysis
from .synthetic import generate_dataset
from .tracing import failure as failure_payload
from .tracing import trace_entry, utc_now_iso
from .versioning import SCHEMA_MIGRATION_VERSION


def error_body(settings: Settings, request_id: str, code: str, message: str, details=None):
    return {
        "schema_version": settings.schema_version,
        "service_version": settings.service_version,
        "request_id": request_id,
        "error": {
            "code": code,
            "message": message,
            "details": details or [],
        },
        "failures": [failure_payload(code, message, "request")],
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.repository = SurvivalRepository(settings.db_path)
        yield

    app = FastAPI(
        title="Synthetic Equipment Survival Analysis API",
        version=settings.service_version,
        lifespan=lifespan,
        description=(
            "Kaplan-Meier and log-rank calculations for local synthetic equipment "
            "lifetime data. Not intended for medical decisions."
        ),
    )

    def repository(request: Request) -> SurvivalRepository:
        return request.app.state.repository

    def resolve_request_id(request: Request) -> str:
        return request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex}"

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = resolve_request_id(request)
        logging.getLogger("api.validation").warning(
            "request validation failed",
            extra={"request_id": request_id},
        )
        details = []
        for item in exc.errors():
            location = ".".join(str(part) for part in item.get("loc", []))
            details.append(
                {
                    "location": location,
                    "type": item.get("type"),
                    "message": item.get("msg"),
                    "failure_category": item.get("type"),
                }
            )
        return JSONResponse(
            status_code=422,
            content=error_body(
                settings,
                request_id,
                "VALIDATION_ERROR",
                "The request did not satisfy the survival analysis contract.",
                details,
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = resolve_request_id(request)
        logging.getLogger("api.error").exception(
            "unhandled request failure",
            extra={"request_id": request_id},
        )
        return JSONResponse(
            status_code=500,
            content=error_body(
                settings,
                request_id,
                "INTERNAL_ERROR",
                f"Unexpected processing failure: {type(exc).__name__}",
                [{"message": str(exc)}],
            ),
        )

    @app.middleware("http")
    async def correlate_and_log(request: Request, call_next):
        started = time.perf_counter()
        started_at = utc_now_iso()
        request_id = resolve_request_id(request)
        request.state.request_id = request_id
        detail = {"method": request.method, "path": request.url.path}
        try:
            repository(request).insert_http_log(
                request_id=request_id,
                started_at=started_at,
                method=request.method,
                path=request.url.path,
                detail=detail,
            )
        except Exception:
            pass

        try:
            response = await call_next(request)
        except Exception:
            try:
                repository(request).finish_http_log(
                    request_id=request_id,
                    finished_at=utc_now_iso(),
                    status_code=500,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                    detail={**detail, "request_id": request_id, "error": True},
                )
            except Exception:
                pass
            raise
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        response.headers["X-Request-ID"] = request_id
        try:
            repository(request).finish_http_log(
                request_id=request_id,
                finished_at=utc_now_iso(),
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
                detail={**detail, "request_id": request_id},
            )
        except Exception:
            pass
        return response

    @app.get("/health")
    async def health(request: Request):
        return {
            "status": "ok",
            "service": settings.app_name,
            "service_version": settings.service_version,
            "schema_version": settings.schema_version,
            "kernel_version": settings.kernel_version,
            "request_id": request.state.request_id,
        }

    @app.post("/api/v1/analyses", response_model=SurvivalAnalysisResponse)
    async def create_analysis(
        payload: SurvivalAnalysisRequest,
        request: Request,
    ):
        request_id = request.state.request_id
        analysis_id = f"km-{uuid.uuid4().hex}"
        result = run_analysis(
            payload,
            settings,
            analysis_id=analysis_id,
            request_id=request_id,
            created_at=utc_now_iso(),
        )
        repo = repository(request)
        result["trace"].append(
            trace_entry(
                "sqlite",
                "analysis_persisted",
                status="ok",
                database=settings.db_path,
                schema_migration=f"user_version={SCHEMA_MIGRATION_VERSION}",
            )
        )
        repo.save_analysis(payload.model_dump(mode="json"), result)
        return result

    @app.get("/api/v1/analyses/{analysis_id}", response_model=SurvivalAnalysisResponse)
    async def read_analysis(analysis_id: str, request: Request):
        try:
            return repository(request).get_analysis(analysis_id)
        except AnalysisNotFound:
            return JSONResponse(
                status_code=404,
                content=error_body(
                    settings,
                    request.state.request_id,
                    "ANALYSIS_NOT_FOUND",
                    f"No analysis exists with analysis_id {analysis_id}.",
                    [{"analysis_id": analysis_id}],
                ),
            )

    @app.get("/api/v1/analyses/{analysis_id}/events")
    async def read_analysis_events(analysis_id: str, request: Request):
        try:
            events = repository(request).get_events(analysis_id)
        except AnalysisNotFound:
            return JSONResponse(
                status_code=404,
                content=error_body(
                    settings,
                    request.state.request_id,
                    "ANALYSIS_NOT_FOUND",
                    f"No analysis exists with analysis_id {analysis_id}.",
                    [{"analysis_id": analysis_id}],
                ),
            )
        return {
            "analysis_id": analysis_id,
            "request_id": request.state.request_id,
            "events": events,
        }

    @app.post("/api/v1/synthetic-dataset")
    async def create_synthetic_dataset(payload: SyntheticDatasetRequest):
        return generate_dataset(payload)

    return app


app = create_app()
