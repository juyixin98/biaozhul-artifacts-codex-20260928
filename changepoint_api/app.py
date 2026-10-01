"""FastAPI application factory and dependency wiring."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .errors import ChangepointError
from .models import (
    DetectionRequest,
    DetectionResponse,
    ErrorBody,
    EventResponse,
    RunStatusResponse,
)
from .repository import SQLiteRepository
from .service import DetectionService
from .versions import (
    ALGORITHM_VERSION,
    KERNEL_VERSION,
    PACKAGE_VERSION,
    PRUNED_KERNEL_VERSION,
)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def get_repository(
    settings: Annotated[Settings, Depends(get_settings)],
) -> SQLiteRepository:
    return SQLiteRepository(settings.db_path)


def get_service(
    settings: Annotated[Settings, Depends(get_settings)],
    repository: Annotated[SQLiteRepository, Depends(get_repository)],
) -> DetectionService:
    return DetectionService(settings, repository)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = FastAPI(
        title="Offline Mean Change-Point Detection",
        version=PACKAGE_VERSION,
        description=(
            "Exactly minimizes segment SSE plus a per-segment penalty."
        ),
    )
    repository = SQLiteRepository(settings.db_path)
    repository.initialize()

    @app.exception_handler(ChangepointError)
    async def changepoint_error_handler(
        request: Request, exc: ChangepointError
    ) -> JSONResponse:
        body = ErrorBody(
            error_code=exc.error_code,
            message=exc.message,
            run_id=exc.run_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        body = ErrorBody(
            error_code="unexpected_error",
            message=str(exc) or "unexpected error without a message",
        )
        return JSONResponse(status_code=500, content=body.model_dump())

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "package_version": PACKAGE_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
        }

    @app.get("/versions")
    def versions() -> dict[str, str]:
        return {
            "package": PACKAGE_VERSION,
            "algorithm": ALGORITHM_VERSION,
            "exact_kernel": KERNEL_VERSION,
            "pruned_kernel": PRUNED_KERNEL_VERSION,
        }

    @app.post(
        "/api/v1/detections",
        response_model=DetectionResponse,
        status_code=201,
        tags=["detection"],
    )
    def create_detection(
        request: DetectionRequest,
        service: Annotated[DetectionService, Depends(get_service)],
    ) -> DetectionResponse:
        return service.run_detection(request)

    @app.get(
        "/api/v1/runs/{run_id}",
        response_model=RunStatusResponse,
        tags=["detection"],
    )
    def get_run(
        run_id: str,
        service: Annotated[DetectionService, Depends(get_service)],
    ) -> RunStatusResponse:
        return service.get_run(run_id)

    @app.get(
        "/api/v1/runs/{run_id}/events",
        response_model=list[EventResponse],
        tags=["detection"],
    )
    def get_run_events(
        run_id: str,
        service: Annotated[DetectionService, Depends(get_service)],
    ):
        service.get_run(run_id)
        return service.get_events(run_id)

    return app


app = create_app()
