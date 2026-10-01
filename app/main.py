from __future__ import annotations

import os
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app.models import CovarianceRequest, CovarianceResponse, HealthResponse
from app.persistence import ensure_database, record_request
from app.sample_contract import SampleContractError
from app.service import estimate
from app.versioning import ESTIMATOR_VERSION, SCHEMA_VERSION

app = FastAPI(
    title="Robust Covariance Estimation API",
    version=ESTIMATOR_VERSION,
    description="Ledoit-Wolf shrinkage and regularized Tyler covariance estimates.",
)


@app.on_event("startup")
def startup() -> None:
    ensure_database()


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    started = time.perf_counter()
    response: Response = await call_next(request)
    response.headers["x-request-id"] = request_id
    response.headers["x-duration-ms"] = (
        f"{(time.perf_counter() - started) * 1000.0:.4f}"
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    db_path = ensure_database()
    return HealthResponse(
        status="ok",
        request_id=request.state.request_id,
        database=str(db_path),
    )


@app.post("/api/v1/covariance/estimate", response_model=CovarianceResponse)
def estimate_covariance(payload: CovarianceRequest, request: Request):
    request_id = request.state.request_id
    started = time.perf_counter()
    try:
        response, _status, failures, uncertainties, duration_ms = estimate(
            payload, request_id=request_id
        )
    except SampleContractError as exc:
        code, _, message = str(exc).partition(":")
        body = {
            "schemaVersion": SCHEMA_VERSION,
            "estimatorVersion": ESTIMATOR_VERSION,
            "requestId": request_id,
            "status": "error",
            "method": payload.method,
            "nSamples": len(payload.data),
            "nDimensions": len(payload.data[0]) if payload.data else 0,
            "failures": [
                {"code": code.strip(), "message": message.strip() or str(exc)}
            ],
            "uncertainties": [],
        }
        record_request(
            request_id=request_id,
            method=payload.method,
            status="error",
            n_samples=body["nSamples"],
            n_dimensions=body["nDimensions"],
            duration_ms=(time.perf_counter() - started) * 1000.0,
            request_payload=payload.model_dump(mode="json", by_alias=True),
            response_payload=body,
            failures=body["failures"],
            uncertainties=[],
        )
        return JSONResponse(body, status_code=422)

    serialized = response.model_dump(mode="json", by_alias=True)
    record_request(
        request_id=request_id,
        method=payload.method,
        status=response.status,
        n_samples=response.n_samples,
        n_dimensions=response.n_dimensions,
        duration_ms=duration_ms,
        request_payload=payload.model_dump(mode="json", by_alias=True),
        response_payload=serialized,
        failures=[item.model_dump() for item in response.failures],
        uncertainties=[item.model_dump() for item in response.uncertainties],
    )
    status_code = 200 if response.status != "error" else 422
    return JSONResponse(serialized, status_code=status_code)
