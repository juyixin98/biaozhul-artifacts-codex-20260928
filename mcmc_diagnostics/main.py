"""FastAPI application for MCMC autocorrelation, ESS, and convergence diagnostics."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import Settings, get_settings
from .kernel import DiagnosticInputError, analyze_chains
from .models import DiagnosticsRequest
from .versioning import (
    PROCESSING_LOCATION,
    ResultRepository,
    SCHEMA_VERSION,
    build_versioned_result,
    new_request_id,
)

logger = logging.getLogger("mcmc_diagnostics")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    repository = ResultRepository(settings.database_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            repository.close()

    app = FastAPI(
        title="MCMC Diagnostics Service",
        version="1.0.0",
        description=(
            "Single- and multi-chain ACF, effective sample size, and R-hat "
            "diagnostics with explicit edge-case reporting."
        ),
        lifespan=lifespan,
    )

    def get_repository() -> ResultRepository:
        return repository

    def log(
        repository: ResultRepository,
        request_id: str,
        stage: str,
        level: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        repository.log_event(request_id, stage, level, message, details)
        logger.log(
            getattr(logging, level.upper(), logging.INFO),
            "request_id=%s stage=%s %s",
            request_id,
            stage,
            message,
            extra={"request_id": request_id, "details": details or {}},
        )

    async def request_id_from_sources(request: Request) -> str:
        request_id = (
            request.headers.get("x-request-id")
            or request.headers.get("x-correlation-id")
        )
        try:
            raw_payload = await request.json()
            if isinstance(raw_payload, dict) and raw_payload.get("request_id"):
                request_id = str(raw_payload["request_id"])[:128]
        except Exception:
            pass
        return request_id or new_request_id()

    def serializable_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in error.items() if key != "ctx"}
            for error in errors
        ]

    @app.exception_handler(ValidationError)
    async def handle_validation_error(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        request_id = await request_id_from_sources(request)
        validation_errors = serializable_validation_errors(exc.errors())
        first_error = validation_errors[0] if validation_errors else {}
        raw_message = first_error.get("msg", "Request validation failed.")
        code = "INVALID_REQUEST"
        for known_code in (
            "EQUAL_LENGTH_REQUIRED",
            "EMPTY_SAMPLES",
            "EMPTY_CHAIN",
        ):
            if known_code in raw_message:
                code = known_code
                break
        log(
            repository,
            request_id,
            "request_contract_failed",
            "WARNING",
            raw_message,
            {"error_code": code, "validation_errors": validation_errors},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "request_id": request_id,
                "schema_version": SCHEMA_VERSION,
                "error_code": code,
                "message": raw_message,
                "details": {"validation_errors": validation_errors},
                "processing_location": PROCESSING_LOCATION,
            },
        )

    @app.exception_handler(DiagnosticInputError)
    async def handle_kernel_input_error(
        request: Request, exc: DiagnosticInputError
    ) -> JSONResponse:
        request_id = await request_id_from_sources(request)
        log(
            repository,
            request_id,
            "diagnostic_failed",
            "WARNING",
            exc.message,
            {"error_code": exc.code},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "request_id": request_id,
                "schema_version": SCHEMA_VERSION,
                "error_code": exc.code,
                "message": exc.message,
                "details": {},
                "processing_location": PROCESSING_LOCATION,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = await request_id_from_sources(request)
        validation_errors = serializable_validation_errors(exc.errors())
        message = "Request body could not be parsed as the documented JSON contract."
        log(
            repository,
            request_id,
            "request_contract_failed",
            "WARNING",
            message,
            {"error_code": "INVALID_JSON", "validation_errors": validation_errors},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "request_id": request_id,
                "schema_version": SCHEMA_VERSION,
                "error_code": "INVALID_JSON",
                "message": message,
                "details": {"validation_errors": validation_errors},
                "processing_location": PROCESSING_LOCATION,
            },
        )

    @app.get("/healthz")
    async def health(repo: ResultRepository = Depends(get_repository)) -> dict[str, str]:
        repo.connection.execute("SELECT 1").fetchone()
        return {"status": "ok", "database": str(settings.database_path)}

    @app.post("/api/v1/diagnostics")
    async def diagnostics(
        payload: dict[str, Any],
        request: Request,
        x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
        repo: ResultRepository = Depends(get_repository),
    ) -> dict[str, Any]:
        # Pydantic is invoked explicitly so its error envelope can include the
        # caller correlation ID and a stable failure category.
        model = DiagnosticsRequest.model_validate(payload)
        request_id = (
            model.request_id
            or x_request_id
            or request.headers.get("x-correlation-id")
            or new_request_id()
        )

        n_chains = len(model.chains)
        n_draws = len(model.chains[0])
        if n_chains > settings.max_chains:
            raise DiagnosticInputError(
                "TOO_MANY_CHAINS",
                f"At most {settings.max_chains} chains are accepted by this configuration.",
            )
        if n_draws > settings.max_draws_per_chain:
            raise DiagnosticInputError(
                "TOO_MANY_DRAWS",
                (
                    f"At most {settings.max_draws_per_chain} draws per chain are accepted "
                    "by this configuration."
                ),
            )

        log(
            repo,
            request_id,
            "received",
            "INFO",
            f"Accepted {n_chains} chains with {n_draws} draws each.",
            {"chain_count": n_chains, "draws_per_chain": n_draws},
        )
        result = analyze_chains(
            model.chains,
            max_acf_lag=model.max_acf_lag,
            default_max_acf_lag=settings.default_max_acf_lag,
            rhat_warning_threshold=settings.rhat_warning_threshold,
            ess_low_fraction=settings.ess_low_fraction,
        )
        envelope = build_versioned_result(result, request_id=request_id)
        if model.persist:
            repo.save(envelope)
            log(
                repo,
                request_id,
                "stored",
                "INFO",
                (
                    f"Stored {SCHEMA_VERSION} result at {PROCESSING_LOCATION}; "
                    f"status={result['status']}."
                ),
                {
                    "schema_version": SCHEMA_VERSION,
                    "service_version": envelope["service_version"],
                    "status": result["status"],
                    "database_path": str(settings.database_path),
                },
            )
        else:
            log(
                repo,
                request_id,
                "completed_unpersisted",
                "INFO",
                f"Computed {SCHEMA_VERSION} at {PROCESSING_LOCATION}; result not stored.",
                {"schema_version": SCHEMA_VERSION, "status": result["status"]},
            )
        return envelope

    @app.get("/api/v1/results/{request_id}")
    async def get_result(
        request_id: str, repo: ResultRepository = Depends(get_repository)
    ) -> dict[str, Any]:
        row = repo.get(request_id)
        if row is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "request_id": request_id,
                    "schema_version": SCHEMA_VERSION,
                    "error_code": "RESULT_NOT_FOUND",
                    "message": "No stored analysis result exists for this request ID.",
                    "details": {},
                    "processing_location": PROCESSING_LOCATION,
                },
            )
        return row

    @app.get("/api/v1/results/{request_id}/events")
    async def get_events(
        request_id: str, repo: ResultRepository = Depends(get_repository)
    ) -> dict[str, Any]:
        events = repo.events(request_id)
        if not events:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "request_id": request_id,
                    "schema_version": SCHEMA_VERSION,
                    "error_code": "EVENTS_NOT_FOUND",
                    "message": "No request events exist for this request ID.",
                    "details": {},
                    "processing_location": PROCESSING_LOCATION,
                },
            )
        return {"request_id": request_id, "events": events}

    return app


app = create_app()
