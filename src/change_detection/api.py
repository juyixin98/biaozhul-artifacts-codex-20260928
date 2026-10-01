from __future__ import annotations

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from . import SCHEMA_VERSION, __version__
from .config import Settings
from .errors import ChangeDetectionError
from .observability import configure_logging
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    HealthResponse,
    RunResponse,
)
from .service import ChangePointService


def get_settings() -> Settings:
    return Settings.from_env()


def get_service(settings: Settings = Depends(get_settings)) -> ChangePointService:
    return ChangePointService(settings)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Offline Mean Change Point API",
        version=__version__,
        description="Minimize segment SSE plus a fixed cost for each segment.",
    )
    app.state.settings = Settings.from_env()
    configure_logging(app.state.settings.log_level, app.state.settings.log_path)

    @app.exception_handler(ChangeDetectionError)
    async def change_detection_exception_handler(
        _request: Request, exc: ChangeDetectionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "category": exc.category,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.get("/health", response_model=HealthResponse)
    def health(
        service: ChangePointService = Depends(get_service),
    ) -> HealthResponse:
        database_ok = service.repository.health()
        return HealthResponse(
            status="ok" if database_ok else "degraded",
            database=database_ok,
            core_version=__version__,
            schema_version=SCHEMA_VERSION,
        )

    @app.post(
        "/api/v1/change-points/analyze",
        response_model=AnalyzeResponse,
        responses={
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def analyze(
        request: AnalyzeRequest,
        service: ChangePointService = Depends(get_service),
    ) -> AnalyzeResponse:
        payload = request.model_dump()
        if len(payload["values"]) > service.settings.max_series_length:
            from .errors import InvalidSeriesError

            raise InvalidSeriesError(
                "series exceeds configured maximum length",
                details={
                    "length": len(payload["values"]),
                    "max_series_length": service.settings.max_series_length,
                },
            )
        return AnalyzeResponse(**service.analyze(payload))

    @app.get("/api/v1/runs/{run_id}", response_model=RunResponse)
    def get_run(
        run_id: str,
        service: ChangePointService = Depends(get_service),
    ) -> RunResponse:
        return RunResponse(**service.get_run(run_id))

    return app


app = create_app()
