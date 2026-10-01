"""FastAPI service entrypoint and routes."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.config import settings
from app.errors import RequestError, validation_error
from app.parsing import parse_candidates, parse_problem
from app.service import (
    DatasetStore,
    build_dataset_payload,
    stateless_dataset,
)

logger = logging.getLogger("nds.api")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        openapi_url=f"{settings.api_prefix}/openapi.json",
        docs_url="/docs",
    )
    store = DatasetStore()
    app.state.store = store

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        logger.info(
            "request.start request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request.end request_id=%s status_code=%s",
            request_id,
            response.status_code,
        )
        return response

    def step(component: str, name: str, detail: str) -> dict[str, str]:
        return {"component": component, "step": name, "detail": detail}

    def ok(request: Request, result: dict[str, Any]) -> dict[str, Any]:
        uncertainties = result.get("diagnostics", {}).get("uncertainties", [])
        return {
            "request_id": request.state.request_id,
            "status": "ok",
            "result": result,
            "warnings": uncertainties,
        }

    @app.exception_handler(RequestError)
    async def request_error_handler(request: Request, exc: RequestError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        logger.warning(
            "request.failed request_id=%s code=%s message=%s",
            request_id,
            exc.code,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "request_id": request_id,
                "status": "error",
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "location": "app.api.request_error_handler",
                },
                "warnings": [],
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id,
                "status": "error",
                "error": {
                    "code": "MALFORMED_REQUEST",
                    "message": "request body or parameters could not be parsed",
                    "details": {"errors": exc.errors()},
                    "location": "app.api.validation_exception_handler",
                },
                "warnings": [],
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        logger.exception("request.error request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "status": "error",
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "unexpected backend failure",
                    "details": {"exception_type": type(exc).__name__},
                    "location": "app.api.unexpected_error_handler",
                },
                "warnings": [],
            },
        )

    @app.get("/health")
    @app.get(f"{settings.api_prefix}/health")
    async def health(request: Request) -> dict[str, Any]:
        return ok(
            request,
            {
                "status": "healthy",
                "api_version": settings.api_version,
                "algorithm_version": settings.algorithm_version,
                "location": "app.api.health",
            },
        )

    @app.post(f"{settings.api_prefix}/sort")
    async def sort_stateless(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        objectives, candidates, tolerance = parse_problem(payload)
        processing = [
            step(
                "app.parsing.parse_problem",
                "parse",
                "validated fixed objective directions and exact finite declared values",
            ),
            step(
                "app.kernel.IncrementalSorter.add_candidates",
                "rank",
                "built exact dominance DAG and longest-path layer numbers",
            ),
            step(
                "app.evidence.build_evidence",
                "verify",
                "compared kernel layers with an independent all-pairs peeling reference",
            ),
        ]
        dataset = stateless_dataset(objectives, candidates, tolerance)
        result = build_dataset_payload(dataset, processing)
        result["operation"] = "stateless_sort"
        return ok(request, result)

    @app.post(f"{settings.api_prefix}/datasets", status_code=201)
    async def create_dataset(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        dataset_id = payload.get("dataset_id")
        if dataset_id is not None and (not isinstance(dataset_id, str) or not dataset_id):
            raise validation_error(
                "INVALID_DATASET_ID",
                "dataset_id must be a non-empty string when provided",
            )
        objectives, candidates, tolerance = parse_problem(payload)
        dataset = store.create(objectives, candidates, tolerance, dataset_id)
        processing = [
            step(
                "app.service.DatasetStore.create",
                "create",
                "stored an in-memory dataset under a request-independent id",
            ),
            step(
                "app.evidence.build_evidence",
                "verify",
                "computed pairwise evidence, duplicates, diagnostics and reference layers",
            ),
        ]
        result = build_dataset_payload(dataset, processing)
        result["operation"] = "create_dataset"
        return ok(request, result)

    @app.get(f"{settings.api_prefix}/datasets/{{dataset_id}}")
    async def get_dataset(dataset_id: str, request: Request) -> dict[str, Any]:
        dataset = store.get(dataset_id)
        processing = [
            step(
                "app.service.DatasetStore.get",
                "read",
                "loaded current in-memory kernel state without mutation",
            ),
            step(
                "app.evidence.build_evidence",
                "verify",
                "recomputed independent reference and numerical diagnostics",
            ),
        ]
        result = build_dataset_payload(dataset, processing)
        result["operation"] = "get_dataset"
        return ok(request, result)

    @app.delete(f"{settings.api_prefix}/datasets/{{dataset_id}}")
    async def delete_dataset(dataset_id: str, request: Request) -> dict[str, Any]:
        store.delete(dataset_id)
        return ok(
            request,
            {
                "dataset_id": dataset_id,
                "deleted": True,
                "processing": [
                    step(
                        "app.service.DatasetStore.delete",
                        "delete_dataset",
                        "removed the complete in-memory dataset",
                    )
                ],
            },
        )

    @app.post(f"{settings.api_prefix}/datasets/{{dataset_id}}/candidates:batchInsert")
    async def insert_candidates(
        dataset_id: str, request: Request, payload: dict[str, Any]
    ) -> dict[str, Any]:
        dataset = store.get(dataset_id)
        objectives = dataset.sorter.objectives
        candidates = parse_candidates(payload.get("candidates"), objectives)
        updated, report = store.insert(dataset_id, candidates)
        processing = [
            step(
                "app.parsing.parse_candidates",
                "validate_batch",
                "validated every incoming candidate before any state mutation",
            ),
            step(
                "app.kernel.IncrementalSorter.add_candidates",
                "incremental_insert",
                "added exact dominance edges and propagated longest-path promotions",
            ),
            step(
                "app.evidence.build_evidence",
                "verify",
                "checked updated ranks against the independent all-pairs reference",
            ),
        ]
        result = build_dataset_payload(updated, processing)
        result.update(
            {
                "operation": "insert_candidates",
                "inserted_ids": list(report.inserted_ids),
                "deleted_id": None,
                "affected_ids": list(report.inserted_ids),
                "rank_changes": [
                    {
                        "id": change.id,
                        "from_rank": change.from_rank,
                        "to_rank": change.to_rank,
                    }
                    for change in report.promotions
                ],
            }
        )
        return ok(request, result)

    @app.delete(f"{settings.api_prefix}/datasets/{{dataset_id}}/candidates/{{candidate_id}}")
    async def delete_candidate(
        dataset_id: str, candidate_id: str, request: Request
    ) -> dict[str, Any]:
        updated, report = store.remove_candidate(dataset_id, candidate_id)
        processing = [
            step(
                "app.kernel.IncrementalSorter.remove_candidate",
                "incremental_delete",
                "found reachable descendants, removed one object identity, and recomputed affected fronts",
            ),
            step(
                "app.evidence.build_evidence",
                "verify",
                "verified layer recovery against the independent all-pairs reference",
            ),
        ]
        result = build_dataset_payload(updated, processing)
        result.update(
            {
                "operation": "delete_candidate",
                "inserted_ids": None,
                "deleted_id": report.deleted_id,
                "affected_ids": list(report.affected_ids),
                "rank_changes": [
                    {
                        "id": change.id,
                        "from_rank": change.from_rank,
                        "to_rank": change.to_rank,
                    }
                    for change in report.rank_changes
                ],
            }
        )
        return ok(request, result)

    return app


app = create_app()
