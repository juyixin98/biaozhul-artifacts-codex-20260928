from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .errors import CalibrationError
from .logging_setup import configure_logging
from .repository import SqliteModelRepository
from .schemas import EvaluateRequest, PredictRequest, ResponseEnvelope, Step, TrainRequest
from .services import CalibrationService, utc_now

logger = logging.getLogger("pav_calibration.api")


def get_repository(request: Request) -> SqliteModelRepository:
    return request.app.state.repository


def get_service(
    repository: SqliteModelRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
) -> CalibrationService:
    return CalibrationService(repository, settings)


def step(module: str, settings: Settings, description: str) -> Step:
    return Step(module=module, version=settings.app_version, description=description)


def success_response(
    *,
    request: Request,
    result: dict[str, Any],
    steps: list[Step],
    warnings: list[dict[str, Any]] | None = None,
    uncertainties: list[dict[str, Any]] | None = None,
    duration_ms: float = 0.0,
) -> ResponseEnvelope:
    envelope = ResponseEnvelope(
        request_id=request.state.request_id,
        status="succeeded",
        result=result,
        warnings=warnings or [],
        uncertainties=uncertainties or [],
        steps=steps,
    )
    repository: SqliteModelRepository = request.app.state.repository
    repository.log_request(
        {
            "request_id": request.state.request_id,
            "method": request.method,
            "path": request.url.path,
            "status": "succeeded",
            "module": steps[-1].module if steps else "api",
            "processing_location": "app.api@local:sqlite:" + repository.path,
            "version": request.app.state.settings.app_version,
            "failure_codes": [],
            "duration_ms": duration_ms,
            "detail": {"warnings": warnings or [], "uncertainties": uncertainties or []},
            "created_at_utc": utc_now(),
        }
    )
    return envelope


def create_app(settings: Settings | None = None) -> FastAPI:
    effective_settings = settings or get_settings()
    configure_logging(effective_settings.log_level)
    app = FastAPI(title=effective_settings.app_name, version=effective_settings.app_version)
    app.state.settings = effective_settings
    app.state.repository = SqliteModelRepository(effective_settings.database_path)
    app.state.repository.initialize()

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        start = time.perf_counter()
        logger.info(
            "request started",
            extra={
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "module_location": "app.api",
            },
        )
        try:
            return await call_next(request)
        finally:
            logger.info(
                "request completed",
                extra={
                    "request_id": request.state.request_id,
                    "duration_ms": round((time.perf_counter() - start) * 1000.0, 3),
                    "module_location": "app.api",
                },
            )

    @app.exception_handler(CalibrationError)
    async def calibration_error_handler(request: Request, exc: CalibrationError):
        failure = {
            "code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "location": exc.__class__.__module__,
        }
        repository: SqliteModelRepository = request.app.state.repository
        repository.log_request(
            {
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "status": "failed",
                "module": "domain",
                "processing_location": "app.api@local:sqlite:" + repository.path,
                "version": request.app.state.settings.app_version,
                "failure_codes": [exc.error_code],
                "detail": failure,
                "created_at_utc": utc_now(),
            }
        )
        logger.warning(
            exc.message,
            extra={
                "request_id": request.state.request_id,
                "failure_code": exc.error_code,
                "status": "failed",
                "module_location": exc.__class__.__module__,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ResponseEnvelope(
                request_id=request.state.request_id,
                status="failed",
                failures=[failure],
                steps=[Step(module="domain", version=effective_settings.app_version, description="domain validation")],
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        failure = {
            "code": "request_schema_invalid",
            "message": "request payload does not match the API schema",
            "details": exc.errors(),
            "location": "app.schemas",
        }
        repository: SqliteModelRepository = request.app.state.repository
        repository.log_request(
            {
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "status": "failed",
                "module": "schemas",
                "processing_location": "app.api@local:sqlite:" + repository.path,
                "version": effective_settings.app_version,
                "failure_codes": ["request_schema_invalid"],
                "detail": failure,
                "created_at_utc": utc_now(),
            }
        )
        return JSONResponse(
            status_code=422,
            content=ResponseEnvelope(
                request_id=request.state.request_id,
                status="failed",
                failures=[failure],
                steps=[Step(module="schemas", version=effective_settings.app_version, description="request parsing")],
            ).model_dump(),
        )

    @app.get("/health")
    async def health(request: Request):
        return success_response(
            request=request,
            result={"status": "ok", "version": effective_settings.app_version},
            steps=[step("api", effective_settings, "health check")],
        )

    @app.post("/v1/models", response_model=ResponseEnvelope)
    async def train_model(
        request: Request,
        payload: TrainRequest,
        service: CalibrationService = Depends(get_service),
    ):
        started = time.perf_counter()
        artifact, warnings = service.train(payload.model_dump(exclude_none=True))
        result = {
            "model_id": artifact.model_id,
            "artifact_schema_version": artifact.artifact_schema_version,
            "score_name": artifact.score_name,
            "target_name": artifact.target_name,
            "extrapolation": artifact.extrapolation,
            "score_min": artifact.score_min,
            "score_max": artifact.score_max,
            "bins": [bin_item.as_dict() for bin_item in artifact.bins],
            "probabilities": list(artifact.probabilities),
            "thresholds": list(artifact.thresholds),
            "merge_events": list(artifact.merge_events),
            "weighted_sse": artifact.weighted_sse,
            "total_weight": artifact.total_weight,
            "input_record_count": artifact.input_record_count,
            "zero_weight_count": artifact.zero_weight_count,
            "training_fingerprint": artifact.training_fingerprint,
        }
        steps = [
            step("validation", effective_settings, "validated and materialized calibration records"),
            step("kernel", effective_settings, "grouped equal scores and applied PAV"),
            step("artifacts", effective_settings, "built versioned piecewise-constant artifact"),
            step("repository", effective_settings, "persisted artifact to local SQLite"),
        ]
        return success_response(
            request=request,
            result=result,
            steps=steps,
            warnings=warnings,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    @app.post("/v1/predictions", response_model=ResponseEnvelope)
    async def create_prediction(
        request: Request,
        payload: PredictRequest,
        service: CalibrationService = Depends(get_service),
    ):
        _, predictions, uncertainties, warnings = service.predict(payload.model_dump(exclude_none=True))
        steps = [
            step("repository", effective_settings, "loaded artifact by model_id"),
            step("prediction", effective_settings, "applied piecewise-constant score mapping"),
        ]
        return success_response(
            request=request,
            result={"predictions": predictions},
            steps=steps,
            warnings=warnings,
            uncertainties=uncertainties,
        )

    @app.post("/v1/evaluations", response_model=ResponseEnvelope)
    async def create_evaluation(
        request: Request,
        payload: EvaluateRequest,
        service: CalibrationService = Depends(get_service),
    ):
        _, predictions, metrics = service.evaluate(payload.model_dump(exclude_none=True))
        steps = [
            step("repository", effective_settings, "loaded artifact by model_id"),
            step("evaluation", effective_settings, "checked dataset separation and computed metrics"),
        ]
        result = {"metrics": metrics, "predictions": predictions}
        uncertainties = []
        if metrics["extrapolated_count"] > 0:
            uncertainties.append(
                {
                    "code": "evaluation_contains_out_of_range_scores",
                    "message": "evaluation used clipped endpoint probabilities for out-of-range scores",
                    "count": metrics["extrapolated_count"],
                }
            )
        return success_response(
            request=request,
            result=result,
            steps=steps,
            uncertainties=uncertainties,
        )

    @app.get("/v1/models", response_model=ResponseEnvelope)
    async def list_models(
        request: Request,
        repository: SqliteModelRepository = Depends(get_repository),
    ):
        return success_response(
            request=request,
            result={"models": repository.list_models()},
            steps=[step("repository", effective_settings, "listed local models")],
        )

    @app.get("/v1/request-logs/{request_id}", response_model=ResponseEnvelope)
    async def get_logs(
        request: Request,
        request_id: str,
        repository: SqliteModelRepository = Depends(get_repository),
    ):
        rows = [dict(row) for row in repository.get_request_logs(request_id)]
        return success_response(
            request=request,
            result={"request_id": request_id, "logs": rows},
            steps=[step("repository", effective_settings, "read correlated request logs")],
        )

    return app
