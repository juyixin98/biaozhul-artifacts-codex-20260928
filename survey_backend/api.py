from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from survey_backend import __version__
from survey_backend.config import Settings, load_settings
from survey_backend.errors import AnalysisError
from survey_backend.models import (
    AnalysisGroupResponse,
    AnalysisRequest,
    AnalysisResponse,
    ErrorResponse,
    HealthResponse,
)
from survey_backend.persistence import ResultStore
from survey_backend.service import AnalysisService
from survey_backend.statistical.privacy import redact_identifier


logger = logging.getLogger("survey_backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    store = ResultStore(settings.database_path)
    app.state.store = store
    app.state.analysis_service = AnalysisService(store, settings, logger)
    try:
        yield
    finally:
        store.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or load_settings()
    app = FastAPI(
        title="Synthetic Survey Imputation API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings

    def get_service(request: Request) -> AnalysisService:
        return request.app.state.analysis_service

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = request.headers.get("x-request-id")
        details: dict[str, str | int | None] = {"error_count": len(exc.errors())}
        if exc.errors():
            first = exc.errors()[0]
            loc = [str(item) for item in first.get("loc", []) if item != "body"]
            details["field"] = ".".join(loc[-2:])
            details["error_type"] = first.get("type", "validation_error")
        logger.info(
            "request rejected request_id=%s code=invalid_contract field=%s",
            redact_identifier(request_id),
            details.get("field"),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error_code="invalid_contract",
                message="Request does not satisfy the synthetic survey contract",
                request_id=request_id,
                redacted_request_id=redact_identifier(request_id),
                details=details,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(AnalysisError)
    async def analysis_exception_handler(request: Request, exc: AnalysisError) -> JSONResponse:
        request_id = request.headers.get("x-request-id")
        logger.info(
            "analysis rejected request_id=%s code=%s",
            redact_identifier(request_id),
            exc.error_code,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error_code=exc.error_code,
                message=str(exc),
                request_id=request_id,
                redacted_request_id=redact_identifier(request_id),
                details={"failure_class": exc.__class__.__name__},
            ).model_dump(mode="json"),
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.post(
        "/api/v1/synthetic/analyses",
        response_model=AnalysisResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_analysis(
        payload: AnalysisRequest,
        request: Request,
        service: AnalysisService = Depends(get_service),
        x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
        x_analysis_id: str | None = Header(default=None, alias="X-Analysis-ID"),
    ) -> AnalysisResponse:
        return service.analyze(
            payload,
            analysis_id=x_analysis_id,
            supplied_request_id=x_request_id,
        )

    @app.get(
        "/api/v1/synthetic/analyses/{analysis_id}/versions/latest",
        response_model=AnalysisResponse,
    )
    def get_latest_analysis(
        analysis_id: str,
        service: AnalysisService = Depends(get_service),
    ) -> AnalysisResponse:
        response = service.store.get(analysis_id)
        if response is None:
            raise HTTPException(status_code=404, detail="analysis version not found")
        return response

    @app.get(
        "/api/v1/synthetic/analyses/{analysis_id}/versions/{version}",
        response_model=AnalysisResponse,
    )
    def get_analysis_version(
        analysis_id: str,
        version: int,
        service: AnalysisService = Depends(get_service),
    ) -> AnalysisResponse:
        response = service.store.get(analysis_id, version)
        if response is None:
            raise HTTPException(status_code=404, detail="analysis version not found")
        return response

    @app.get(
        "/api/v1/synthetic/analyses/{analysis_id}/versions",
        response_model=AnalysisGroupResponse,
    )
    def list_analysis_versions(
        analysis_id: str,
        service: AnalysisService = Depends(get_service),
    ) -> AnalysisGroupResponse:
        response = service.store.list_versions(analysis_id)
        if response is None:
            raise HTTPException(status_code=404, detail="analysis not found")
        return response

    return app


app = create_app()
