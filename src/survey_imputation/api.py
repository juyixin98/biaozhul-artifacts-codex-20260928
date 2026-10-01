"""FastAPI application exposing generation, analysis, and stored results."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import get_settings
from .contracts import (
    AnalysisRequest,
    AnalysisResponse,
    GeneratedDatasetResponse,
    GenerateDatasetRequest,
)
from .errors import ResultAlreadyExistsError, ResultNotFoundError, SurveyImputationError
from .repository import SQLiteResultRepository
from .service import RejectedAnalysis, SurveyService
from .versioning import CONTRACT_VERSION, CORE_VERSION, STORAGE_VERSION


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Synthetic Survey Imputation Comparison",
        version=CORE_VERSION,
        description="Local synthetic case comparing mean imputation and multiple imputation.",
    )
    app.state.settings = settings
    app.state.repository = SQLiteResultRepository(settings.database_path)
    app.state.service = SurveyService(redact_sensitive=settings.redact_sensitive)

    @app.exception_handler(RejectedAnalysis)
    async def rejected_analysis_handler(request: Request, exc: RejectedAnalysis) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "category": exc.category,
                    "message": str(exc),
                    "request_path": request.url.path,
                },
                "diagnostics": [item.model_dump(mode="json") for item in exc.diagnostics],
            },
        )

    @app.exception_handler(SurveyImputationError)
    async def domain_error_handler(request: Request, exc: SurveyImputationError) -> JSONResponse:
        status_code = (
            409
            if isinstance(exc, ResultAlreadyExistsError)
            else 404
            if isinstance(exc, ResultNotFoundError)
            else 422
        )
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "category": exc.category,
                    "message": str(exc),
                    "request_path": request.url.path,
                }
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "core_version": CORE_VERSION, "contract_version": CONTRACT_VERSION}

    @app.get("/versions")
    async def versions() -> dict[str, str]:
        return {
            "contract": CONTRACT_VERSION,
            "core": CORE_VERSION,
            "storage": STORAGE_VERSION,
        }

    @app.post("/datasets/generate", response_model=GeneratedDatasetResponse)
    async def generate_dataset(request: GenerateDatasetRequest) -> GeneratedDatasetResponse:
        return app.state.service.generate(request)

    @app.post("/analyses", response_model=AnalysisResponse, status_code=201)
    async def create_analysis(request: AnalysisRequest) -> AnalysisResponse:
        response = app.state.service.analyze(request)
        app.state.repository.save_response(response)
        return response

    @app.get("/analyses/{request_id}", response_model=AnalysisResponse)
    async def get_analysis(request_id: str) -> AnalysisResponse:
        return app.state.repository.get_response(request_id).response

    return app


app = create_app()
