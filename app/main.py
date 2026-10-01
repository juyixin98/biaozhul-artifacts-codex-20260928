from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .mcmc_diagnostics.config import Settings
from .mcmc_diagnostics.dependencies import cached_settings, get_logger, get_repository
from .mcmc_diagnostics.kernel import analyze_chains, analyze_single_chain
from .mcmc_diagnostics.logging_config import log_event
from .mcmc_diagnostics.models import (
    MultiChainRequest,
    MultiChainResult,
    SingleChainRequest,
    SingleChainResult,
)
from .mcmc_diagnostics.repository import DiagnosticRepository
from .mcmc_diagnostics.versioning import request_id_from_payload, utc_now_iso, version_payload


def create_app() -> FastAPI:
    settings = cached_settings()
    logger = get_logger()
    app = FastAPI(
        title="MCMC Diagnostics Service",
        version=settings.api_version,
        description="Autocorrelation, ESS, and convergence diagnostics for local synthetic samples.",
    )

    def error_body(
        request_id: str,
        code: str,
        message: str,
        details: Any | None = None,
        http_status: int = 400,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=http_status,
            content={
                "request_id": request_id,
                "status": "error",
                "error": {
                    "code": code,
                    "message": message,
                    "details": details or [],
                },
                "uncertainties": [
                    "No statistical diagnosis was produced for the failed request; "
                    "the failure must not be interpreted as evidence of convergence."
                ],
                "versions": version_payload(settings),
                "created_at": utc_now_iso(),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        payload = await _safe_json(request)
        provided_id = (
            payload.get("request_id")
            if isinstance(payload, dict) and isinstance(payload.get("request_id"), str)
            else None
        )
        request_id = request_id_from_payload(payload, provided_id)
        details = []
        for error in exc.errors():
            details.append(
                {
                    "location": ".".join(str(item) for item in error.get("loc", [])),
                    "code": error.get("type", "validation_error"),
                    "message": error.get("msg", "invalid request"),
                }
            )
        code = "SAMPLE_CONTRACT_VIOLATION"
        message = "Request does not satisfy the finite numeric sample contract."
        row_id = None
        if settings.persist_results:
            row_id = get_repository().save_failure(
                str(request.url.path), request_id, code, message
            )
        log_event(
            logger,
            request_id,
            "request_failed",
            logging.WARNING,
            endpoint=request.url.path,
            code=code,
            details=details,
            sqlite_row_id=row_id,
        )
        response = error_body(request_id, code, message, details, 422)
        _decorate(response, request_id, row_id, settings)
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        payload = await _safe_json(request)
        request_id = request_id_from_payload(payload, None)
        code = "INVALID_SAMPLE_SHAPE"
        row_id = None
        if settings.persist_results:
            row_id = get_repository().save_failure(
                str(request.url.path), request_id, code, str(exc)
            )
        log_event(
            logger,
            request_id,
            "request_failed",
            logging.WARNING,
            endpoint=request.url.path,
            code=code,
            message=str(exc),
            sqlite_row_id=row_id,
        )
        response = error_body(request_id, code, str(exc), None, 422)
        _decorate(response, request_id, row_id, settings)
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.service_name}

    @app.get("/version")
    async def version() -> dict[str, str]:
        return version_payload(settings)

    @app.post("/api/v1/diagnostics/single", response_model=SingleChainResult)
    async def single_diagnostics(
        request_model: SingleChainRequest,
        request: Request,
        response: Response,
        repo: DiagnosticRepository = Depends(get_repository),
    ) -> SingleChainResult:
        payload = await _safe_json(request)
        result = analyze_single_chain(request_model, settings, payload=payload)
        data = result.model_dump()
        row_id = _persist(
            repo, settings, logger, request_id=result.request_id,
            endpoint="/api/v1/diagnostics/single", result=data,
        )
        _decorate(response, result.request_id, row_id, settings)
        return result

    @app.post("/api/v1/diagnostics/multi", response_model=MultiChainResult)
    async def multi_diagnostics(
        request_model: MultiChainRequest,
        request: Request,
        response: Response,
        repo: DiagnosticRepository = Depends(get_repository),
    ) -> MultiChainResult:
        payload = await _safe_json(request)
        result = analyze_chains(request_model, settings, payload=payload)
        data = result.model_dump()
        row_id = _persist(
            repo, settings, logger, request_id=result.request_id,
            endpoint="/api/v1/diagnostics/multi", result=data,
        )
        _decorate(response, result.request_id, row_id, settings)
        return result

    @app.get("/api/v1/runs/{request_id}")
    async def get_run(
        request_id: str, repo: DiagnosticRepository = Depends(get_repository)
    ) -> dict[str, Any]:
        row = repo.get(request_id)
        if row is None:
            return JSONResponse(
                status_code=404,
                content={
                    "request_id": request_id,
                    "status": "error",
                    "error": {"code": "RUN_NOT_FOUND", "message": "Unknown request_id."},
                },
            )
        return row

    return app


async def _safe_json(request: Request) -> dict[str, Any] | list[Any] | None:
    try:
        return await request.json()
    except Exception:
        return None

def _persist(
    repo: DiagnosticRepository,
    settings: Settings,
    logger: logging.Logger,
    request_id: str,
    endpoint: str,
    result: dict[str, Any],
) -> int | None:
    row_id = None
    if settings.persist_results:
        row_id = repo.save_success(endpoint, result)
    log_event(
        logger,
        request_id,
        "diagnostics_completed",
        logging.INFO,
        endpoint=endpoint,
        status=result.get("status"),
        n_chains=result.get("n_chains"),
        total_draws=result.get("total_draws"),
        diagnostic_count=len(result.get("diagnostics", [])),
        kernel_version=settings.kernel_version,
        sqlite_row_id=row_id,
        database=str(settings.database_path),
    )
    return row_id


def _decorate(
    response: Response, request_id: str, row_id: int | None, settings: Settings
) -> None:
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Kernel-Version"] = settings.kernel_version
    if row_id is not None:
        response.headers["X-SQLite-Row-Id"] = str(row_id)
        response.headers["X-SQLite-Path"] = str(settings.database_path)


app = create_app()
