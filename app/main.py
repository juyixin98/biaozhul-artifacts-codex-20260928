"""FastAPI application entry point."""

from __future__ import annotations

import platform
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request as StarletteRequest

from . import __version__
from .config import get_settings
from .db import initialize_database, insert_request_log
from .logging_setup import configure_logging, get_logger, request_id_context
from .survival.errors import SurvivalValidationError
from .survival.experiments import run_reproduction
from .survival.fixtures import FIXTURES, get_fixture
from .survival.km import kaplan_meier
from .survival.logrank import logrank_from_request
from .survival.models import KMRequest, LogRankRequest
from .survival.synthetic import SyntheticDatasetRequest, generate_synthetic_dataset
from .survival.versioning import ProcessingStep, build_response, resolve_request_id
from .versions import DISCLAIMER, KM_ESTIMATOR_VERSION, LOGRANK_VERSION


settings = get_settings()
processing_location = f"{platform.node()}:{__version__}"
steps_store: dict[str, list[ProcessingStep]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    initialize_database(settings.sqlite_path)
    get_logger().info("survival API initialized at %s", processing_location)
    yield


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description=(
        "Synthetic Kaplan-Meier and log-rank backend. Demonstration only; "
        "not intended for real medical conclusions."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID")
    payload_request = request
    body_request_id = None
    if request.method in {"POST", "PUT", "PATCH"}:
        body = await request.body()

        async def receive_body():
            return {"type": "http.request", "body": body, "more_body": False}

        payload_request = StarletteRequest(request.scope, receive_body)
        try:
            payload = json.loads(body or b"{}")
            body_request_id = payload.get("request_id") if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            body_request_id = None

    request_id = resolve_request_id(body_request_id or incoming)
    token = request_id_context.set(request_id)
    request.state.request_id = request_id
    request.state.steps = []
    steps_store[request_id] = request.state.steps
    try:
        response = await call_next(payload_request)
    finally:
        request_id_context.reset(token)

    response.headers["X-Request-ID"] = request_id
    if getattr(request.state, "audit", True):
        insert_request_log(
            sqlite_path=settings.sqlite_path,
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            processing_location=processing_location,
            estimator_versions=getattr(request.state, "estimator_versions", []),
            processing_steps=[step.model_dump() for step in request.state.steps],
            error_category=getattr(request.state, "error_category", None),
            message=getattr(request.state, "error_message", None),
        )
    steps_store.pop(request_id, None)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", resolve_request_id(None))
    details = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", []))
        details.append(
            {
                "category": "request_validation",
                "location": location,
                "reason": error.get("msg", "invalid request"),
            }
        )
    request.state.error_category = "request_validation"
    request.state.error_message = "Request payload failed contract validation."
    get_logger().warning("%s: %s", request.state.error_message, details)
    return JSONResponse(
        status_code=422,
        content={
            "request_id": request_id,
            "failure": {
                "code": "request_validation",
                "message": "Request payload failed contract validation.",
                "reasons": details,
            },
            "uncertain_conclusions": [],
            "disclaimer": DISCLAIMER,
        },
    )


@app.exception_handler(SurvivalValidationError)
async def survival_validation_handler(request: Request, exc: SurvivalValidationError):
    request_id = getattr(request.state, "request_id", resolve_request_id(None))
    request.state.error_category = exc.code
    request.state.error_message = str(exc)
    get_logger().warning("statistical input rejected: %s", exc)
    return JSONResponse(
        status_code=422,
        content={
            "request_id": request_id,
            "failure": {
                "code": exc.code,
                "message": str(exc),
                "reasons": [],
            },
            "uncertain_conclusions": [],
            "disclaimer": DISCLAIMER,
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", resolve_request_id(None))
    request.state.error_category = "internal_error"
    request.state.error_message = str(exc)
    get_logger().exception("unexpected processing failure")
    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "failure": {
                "code": "internal_error",
                "message": "Unexpected server error. See server log and request_id.",
                "reasons": [{"category": "internal_error", "location": "server", "reason": str(exc)}],
            },
            "uncertain_conclusions": [],
            "disclaimer": DISCLAIMER,
        },
    )


@app.get("/health")
async def health(request: Request):
    request.state.audit = False
    return {
        "status": "ok",
        "api_version": __version__,
        "request_id": request.state.request_id,
        "processing_location": processing_location,
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/v1/km")
async def kaplan_meier_endpoint(request: Request, payload: KMRequest):
    request_id = resolve_request_id(payload.request_id or request.state.request_id)
    request.state.request_id = request_id
    steps = request.state.steps
    steps.extend(
        [
            ProcessingStep(
                step="parse-contract",
                status="completed",
                detail=f"Accepted {len(payload.observations)} observations.",
            ),
            ProcessingStep(
                step="risk-set-order",
                status="declared",
                detail=(
                    "Risk set is entry_time < t <= time. Same-time delayed "
                    "entrants are shown at t and enter after same-time exits."
                ),
            ),
            ProcessingStep(
                step="fit-kaplan-meier",
                status="completed",
                detail=(
                    f"Method={payload.confidence_interval_method}, "
                    f"confidence={payload.confidence_level:.3f}, "
                    "Greenwood variance with explicit boundary handling."
                ),
            ),
        ]
    )
    get_logger().info(
        "fitting KM for %d observations, method=%s",
        len(payload.observations),
        payload.confidence_interval_method,
    )
    result = kaplan_meier(
        payload.observations,
        confidence_level=payload.confidence_level,
        confidence_interval_method=payload.confidence_interval_method,
        include_censor_times=payload.include_censor_times,
        include_entry_times=payload.include_entry_times,
    )
    request.state.estimator_versions = [KM_ESTIMATOR_VERSION]
    response = build_response(
        result=result,
        request_id=request_id,
        steps=steps,
        processing_location=processing_location,
        estimator_versions=[KM_ESTIMATOR_VERSION],
    )
    return response


@app.post("/api/v1/logrank")
async def logrank_endpoint(request: Request, payload: LogRankRequest):
    request_id = resolve_request_id(payload.request_id or request.state.request_id)
    request.state.request_id = request_id
    steps = request.state.steps
    steps.extend(
        [
            ProcessingStep(
                step="parse-contract",
                status="completed",
                detail=f"Accepted {len(payload.observations)} observations.",
            ),
            ProcessingStep(
                step="risk-set-order",
                status="declared",
                detail=(
                    "Breslow tied-event ordering; pre-event risk set is "
                    "entry_time < t <= time, including same-time censored subjects."
                ),
            ),
            ProcessingStep(
                step="fit-log-rank",
                status="completed",
                detail="Omnibus chi-square comparison with observed-minus-expected contributions.",
            ),
        ]
    )
    get_logger().info("fitting log-rank for %d observations", len(payload.observations))
    result = logrank_from_request(payload)
    request.state.estimator_versions = [LOGRANK_VERSION]
    return build_response(
        result=result,
        request_id=request_id,
        steps=steps,
        processing_location=processing_location,
        estimator_versions=[LOGRANK_VERSION],
    )


@app.post("/api/v1/synthetic-dataset")
async def synthetic_dataset_endpoint(payload: SyntheticDatasetRequest):
    return generate_synthetic_dataset(payload)


@app.get("/api/v1/fixtures/{name}")
async def fixture_endpoint(name: str):
    records = get_fixture(name)
    return {
        "fixture_version": "fixtures-v1",
        "name": name,
        "observations": [record.model_dump(mode="json") for record in records],
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/v1/fixtures")
async def list_fixtures_endpoint():
    return {
        "fixture_version": "fixtures-v1",
        "fixtures": sorted(FIXTURES),
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/v1/reproduce")
async def reproduce_endpoint():
    return run_reproduction()
