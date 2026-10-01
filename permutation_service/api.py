"""FastAPI application factory and HTTP boundary."""


from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import Settings
from .errors import PermutationError
from .schemas import (
    CancelResponse,
    CreateRunRequest,
    ErrorResponse,
    RunEventResponse,
    RunResponse,
    VersionResponse,
)
from .service import JobRunner
from .storage import RunStore
from .versions import runtime_versions


def _error_payload(
    code: str, message: str, category: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "category": category,
            "details": details or {},
        }
    }


def _status_for_run(run: dict[str, Any]) -> int:
    if run["status"] == "completed":
        return status.HTTP_200_OK
    if run["status"] in {"pending", "running"}:
        return status.HTTP_202_ACCEPTED
    if run["status"] == "canceled":
        return status.HTTP_409_CONFLICT
    category = (run.get("error") or {}).get("category")
    if category in {"invalid_input", "invalid_request"}:
        return status.HTTP_400_BAD_REQUEST
    if category == "not_found":
        return status.HTTP_404_NOT_FOUND
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format=(
            "%(asctime)s %(levelname)s %(name)s "
            "%(filename)s:%(lineno)d %(message)s"
        ),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = RunStore(settings.db_path)
        runner = JobRunner(store, settings)
        runner.start()
        app.state.settings = settings
        app.state.store = store
        app.state.runner = runner
        try:
            yield
        finally:
            runner.stop()

    app = FastAPI(
        title="Exact Permutation Test Service",
        version=runtime_versions()["service"],
        lifespan=lifespan,
    )

    def get_runner(request: Request) -> JobRunner:
        return request.app.state.runner

    @app.exception_handler(PermutationError)
    async def permutation_error_handler(
        request: Request, exc: PermutationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                exc.error_code, exc.message, exc.category, exc.details
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_payload(
                "INVALID_REQUEST",
                "request payload does not satisfy the API schema",
                "invalid_request",
                {"validation_errors": exc.errors()},
            ),
        )

    @app.exception_handler(Exception)
    async def unknown_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger("permutation_service").exception(
            "unhandled HTTP error on %s", request.url.path
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload(
                "UNKNOWN_ERROR",
                "an unexpected server error occurred",
                "unknown",
            ),
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version", response_model=VersionResponse)
    def version() -> dict[str, str]:
        return runtime_versions()

    @app.post(
        "/api/v1/runs",
        response_model=RunResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses={400: {"model": ErrorResponse}},
    )
    def create_run(
        payload: CreateRunRequest,
        runner: JobRunner = Depends(get_runner),
    ) -> dict[str, Any]:
        run_input = payload.model_dump(exclude_none=True)
        return runner.create_run(run_input)

    @app.get(
        "/api/v1/runs/{run_id}",
        response_model=RunResponse,
        responses={
            202: {"model": RunResponse},
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def get_run(
        run_id: str,
        runner: JobRunner = Depends(get_runner),
    ) -> JSONResponse:
        run = runner.get_run(run_id)
        return JSONResponse(status_code=_status_for_run(run), content=run)

    @app.post(
        "/api/v1/runs/{run_id}/cancel",
        response_model=CancelResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def cancel_run(
        run_id: str,
        runner: JobRunner = Depends(get_runner),
    ) -> JSONResponse:
        run = runner.cancel_run(run_id)
        if run["status"] == "completed":
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "run_id": run_id,
                    "status": run["status"],
                    "cancel_requested": run["cancel_requested"],
                    "message": "run already completed and cannot be canceled",
                },
            )
        if run["status"] == "failed":
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "run_id": run_id,
                    "status": run["status"],
                    "cancel_requested": run["cancel_requested"],
                    "message": "run already failed and cannot be canceled",
                },
            )
        if run["status"] == "canceled":
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "run_id": run_id,
                    "status": "canceled",
                    "cancel_requested": True,
                    "message": "run is already canceled",
                },
            )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "run_id": run_id,
                "status": run["status"],
                "cancel_requested": True,
                "message": "cancellation requested",
            },
        )

    @app.get(
        "/api/v1/runs/{run_id}/events",
        response_model=list[RunEventResponse],
        responses={404: {"model": ErrorResponse}},
    )
    def list_events(
        run_id: str,
        runner: JobRunner = Depends(get_runner),
    ) -> list[dict[str, Any]]:
        return runner.list_events(run_id)

    return app


app = create_app()
