"""FastAPI service entry point."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import config
from app.audit import pairwise_relation
from app.errors import NsgaError
from app.models import CandidateBatch, CreateProblemRequest
from app.service import registry, response, state_view
from app.core import candidate_relations


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        if not hasattr(record, "stage"):
            record.stage = "-"
        return True


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(config.SERVICE_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s request_id=%(request_id)s "
                "stage=%(stage)s %(message)s"
            )
        )
        handler.addFilter(RequestIdFilter())
        logger.addHandler(handler)
    logger.setLevel(config.DEFAULT_LOG_LEVEL)
    return logger


logger = configure_logging()


def _request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID", f"req-{uuid.uuid4().hex[:12]}")


def _log(level: int, request_id: str, stage: str, message: str) -> None:
    logger.log(
        level,
        message,
        extra={"request_id": request_id, "stage": stage},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log(
        logging.INFO,
        "startup",
        "lifecycle",
        f"{config.SERVICE_NAME} {config.SERVICE_VERSION} starting",
    )
    yield
    _log(
        logging.INFO,
        "shutdown",
        "lifecycle",
        f"{config.SERVICE_NAME} {config.SERVICE_VERSION} stopping",
    )


app = FastAPI(
    title="Exact Finite-Set NSGA Backend",
    version=config.SERVICE_VERSION,
    lifespan=lifespan,
)


@app.exception_handler(NsgaError)
async def nsga_error_handler(request: Request, exc: NsgaError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", _request_id(request))
    error = {
        "code": exc.code,
        "message": exc.message,
        "stage": exc.stage,
        "details": exc.details,
    }
    _log(
        logging.WARNING,
        request_id,
        exc.stage,
        f"failure={exc.code} {exc.message}",
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": request_id,
            "version": config.SERVICE_VERSION,
            "stage": exc.stage,
            "data": None,
            "evidence": None,
            "errors": [error],
            "uncertainty": [],
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", _request_id(request))
    errors = [
        {
            "code": "schema_validation_failed",
            "message": error.get("msg", "Request schema validation failed."),
            "stage": "request-schema",
            "details": {
                "location": list(error.get("loc", [])),
                "type": error.get("type"),
            },
        }
        for error in exc.errors()
    ]
    _log(logging.WARNING, request_id, "request-schema", f"{len(errors)} schema error(s)")
    return JSONResponse(
        status_code=422,
        content={
            "request_id": request_id,
            "version": config.SERVICE_VERSION,
            "stage": "request-schema",
            "data": None,
            "evidence": None,
            "errors": errors,
            "uncertainty": [],
        },
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = _request_id(request)
    request.state.request_id = request_id
    _log(
        logging.INFO,
        request_id,
        "request-received",
        f"{request.method} {request.url.path}",
    )
    response_object = await call_next(request)
    response_object.headers["X-Request-ID"] = request_id
    response_object.headers["X-Service-Version"] = config.SERVICE_VERSION
    return response_object


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    request_id = request.state.request_id
    _log(logging.INFO, request_id, "health", f"algorithm={config.ALGORITHM_VERSION}")
    return {
        "request_id": request_id,
        "version": config.SERVICE_VERSION,
        "stage": "health",
        "algorithm_version": config.ALGORITHM_VERSION,
        "status": "ok",
    }


@app.post("/v1/problems")
def create_problem(request: Request, payload: CreateProblemRequest) -> dict[str, Any]:
    request_id = request.state.request_id
    _log(logging.INFO, request_id, "model-parsing", "creating problem")
    store, evidence = registry.create(payload, request_id)
    _log(logging.INFO, request_id, "response", f"created problem={store.problem_id}")
    return response(
        request_id=request_id,
        stage="problem-created",
        data={"state": state_view(store)},
        store=store,
        evidence=evidence,
    )


@app.get("/v1/problems/{problem_id}")
def read_problem(request: Request, problem_id: str) -> dict[str, Any]:
    request_id = request.state.request_id
    store = registry.get(problem_id)
    _log(
        logging.INFO,
        request_id,
        "state-read",
        f"read problem={problem_id} candidates={len(store.points)}",
    )
    return response(
        request_id=request_id,
        stage="state-read",
        data={"state": state_view(store)},
        store=store,
    )


@app.post("/v1/problems/{problem_id}/candidates")
def insert_candidate(
    request: Request, problem_id: str, payload: CandidateBatch
) -> dict[str, Any]:
    request_id = request.state.request_id
    candidates = payload.candidates if payload.candidates is not None else []
    _log(logging.INFO, request_id, "model-parsing", f"insert batch size={len(candidates)}")
    store, evidence = registry.add(problem_id, candidates, request_id)
    candidate_id = evidence.inserted_candidate_ids[0] if evidence.inserted_candidate_ids else ""
    _log(logging.INFO, request_id, evidence.stage, f"affected={len(evidence.affected_candidate_ids)}")
    return response(
        request_id=request_id,
        stage=evidence.stage,
        data={"candidate_id": candidate_id, "state": state_view(store)},
        store=store,
        evidence=evidence,
    )


@app.delete("/v1/problems/{problem_id}/candidates/{candidate_id}")
def remove_candidate(
    request: Request, problem_id: str, candidate_id: str
) -> dict[str, Any]:
    request_id = request.state.request_id
    _log(logging.INFO, request_id, "candidate-delete", f"candidate={candidate_id}")
    store, evidence = registry.delete(problem_id, candidate_id, request_id)
    _log(logging.INFO, request_id, evidence.stage, f"edges_removed={evidence.edges_removed}")
    return response(
        request_id=request_id,
        stage=evidence.stage,
        data={"candidate_id": candidate_id, "state": state_view(store)},
        store=store,
        evidence=evidence,
    )


@app.get("/v1/problems/{problem_id}/candidates/{candidate_id}")
def read_candidate(
    request: Request, problem_id: str, candidate_id: str
) -> dict[str, Any]:
    request_id = request.state.request_id
    store = registry.get(problem_id)
    record, dominates_ids, dominated_by_ids, duplicate_ids = candidate_relations(
        store, candidate_id
    )
    _log(logging.INFO, request_id, "candidate-read", f"candidate={candidate_id} rank={record.rank}")
    return response(
        request_id=request_id,
        stage="candidate-read",
        data={
            "candidate_id": candidate_id,
            "rank": record.rank,
            "duplicate_group": record.group_id,
            "duplicates": duplicate_ids,
            "dominates": dominates_ids,
            "dominated_by": dominated_by_ids,
        },
        store=store,
    )


@app.get("/v1/problems/{problem_id}/explain/{left_id}/{right_id}")
def explain_pair(
    request: Request,
    problem_id: str,
    left_id: str,
    right_id: str,
) -> dict[str, Any]:
    request_id = request.state.request_id
    store = registry.get(problem_id)
    candidate_relations(store, left_id)
    candidate_relations(store, right_id)
    left = store.points[left_id]
    right = store.points[right_id]
    relation = pairwise_relation(left, right)
    per_dimension = []
    for index, (name, direction) in enumerate(zip(store.objective_names, store.directions, strict=True)):
        per_dimension.append(
            {
                "index": index,
                "name": name,
                "direction": direction,
                "left": str(left.canonical[index]),
                "right": str(right.canonical[index]),
                "comparison": (
                    "equal"
                    if left.canonical[index] == right.canonical[index]
                    else "left_better"
                    if left.canonical[index] > right.canonical[index]
                    else "right_better"
                ),
            }
        )
    _log(logging.INFO, request_id, "evidence", f"relation={relation}")
    return response(
        request_id=request_id,
        stage="pairwise-explanation",
        data={
            "left_id": left_id,
            "right_id": right_id,
            "relation": relation,
            "strict_dominance_rule": "all objectives not worse and at least one better",
            "dimensions": per_dimension,
        },
        store=store,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.DEFAULT_HOST,
        port=config.DEFAULT_PORT,
        log_level=config.DEFAULT_LOG_LEVEL.lower(),
    )
