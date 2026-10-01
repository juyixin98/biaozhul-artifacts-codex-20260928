"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .kernel import InvalidSeriesError
from .repository import (
    AnalysisNotFoundError,
    AnalysisRepository,
    DuplicateRequestError,
)
from .schemas import (
    AnalysisRequestPayload,
    AnalysisResponsePayload,
    ErrorPayload,
)
from .service import run_analysis, serialize_outcome


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    repository = AnalysisRepository(settings.database_path)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        repository.initialize()
        yield

    app = FastAPI(
        title="Irregular Lag Correlation API",
        version="1.0.0",
        description=(
            "One-to-one tolerance-window matching and fixed Pearson lag "
            "correlation for irregularly sampled sequences."
        ),
        lifespan=lifespan,
    )

    def get_repository() -> AnalysisRepository:
        return repository

    @app.exception_handler(InvalidSeriesError)
    def invalid_series_handler(_request: Request, exc: InvalidSeriesError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorPayload(
                error_code="INVALID_SERIES",
                message=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(DuplicateRequestError)
    def duplicate_handler(_request: Request, exc: DuplicateRequestError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=ErrorPayload(
                error_code="DUPLICATE_REQUEST_ID",
                message="request_id already exists",
                request_id=exc.request_id,
            ).model_dump(),
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/analyses",
        response_model=AnalysisResponsePayload,
        status_code=201,
    )
    def create_analysis(
        payload: AnalysisRequestPayload,
        repo: AnalysisRepository = Depends(get_repository),
    ) -> dict:
        request_id, outcome, status_summary = run_analysis(payload, repo)
        return serialize_outcome(
            request_id,
            outcome,
            status_summary,
            include_pairs=payload.include_pairs,
        )

    @app.get("/analyses/{request_id}", response_model=AnalysisResponsePayload)
    def get_analysis(
        request_id: str,
        repo: AnalysisRepository = Depends(get_repository),
    ) -> dict:
        try:
            return repo.get_analysis(request_id)
        except AnalysisNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "ANALYSIS_NOT_FOUND",
                    "message": "no analysis exists for this request_id",
                    "request_id": request_id,
                },
            ) from exc

    return app


app = create_app()
