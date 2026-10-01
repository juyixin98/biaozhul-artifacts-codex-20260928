"""FastAPI entry point for calibration analysis."""

from __future__ import annotations

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from core import __version__
from core.config import Settings
from core.errors import CalibrationError, RunNotFoundError
from core.logging_utils import JsonLineLogger
from core.repository import ALGORITHM_VERSION, SCHEMA_VERSION, RunRepository
from core.service import CalibrationService


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    repository = RunRepository(resolved_settings.database_path)
    logger = JsonLineLogger(resolved_settings.log_path)
    service = CalibrationService(repository, logger)

    app = FastAPI(
        title="Probability Calibration Lab",
        version=__version__,
        description="Fixed and equal-frequency weighted calibration analysis with Brier decomposition.",
    )
    app.state.settings = resolved_settings
    app.state.service = service

    def get_service() -> CalibrationService:
        return app.state.service

    @app.exception_handler(CalibrationError)
    async def handle_calibration_error(_: Request, exc: CalibrationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "error_category": exc.category,
                "message": str(exc),
            },
        )

    @app.exception_handler(RunNotFoundError)
    async def handle_run_not_found(_: Request, exc: RunNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "status": "failed",
                "error_category": exc.category,
                "message": str(exc),
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "app_version": __version__}

    @app.get("/version")
    async def version() -> dict[str, object]:
        return {
            "app_version": __version__,
            "algorithm_version": ALGORITHM_VERSION,
            "schema_version": SCHEMA_VERSION,
        }

    @app.post("/api/v1/analyses")
    async def create_analysis(
        request: dict,
        calibration_service: CalibrationService = Depends(get_service),
    ):
        return calibration_service.analyze_raw(request)

    @app.post("/api/v1/calibrations")
    async def fit_calibration(
        request: dict,
        calibration_service: CalibrationService = Depends(get_service),
    ):
        return calibration_service.fit_calibration_raw(request)

    @app.get("/api/v1/runs/{run_id}")
    async def get_run(run_id: str, calibration_service: CalibrationService = Depends(get_service)):
        return calibration_service.get_run(run_id)

    return app


app = create_app()
