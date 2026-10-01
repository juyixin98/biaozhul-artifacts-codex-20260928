"""FastAPI surface: request validation, run orchestration, persistence."""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .config import Settings, load_settings
from .contracts import ErrorResponse, PermutationTestRequest, PermutationTestResponse
from .errors import DomainError, ErrorCode
from .kernel import KERNEL_VERSION, STATUS_CANCELLED, library_versions, permutation_test
from .logging_utils import get_logger, log_event
from .storage import RunStore

API_VERSION = "v1"

logger = get_logger("permtest.service")


def _input_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_app(settings: Settings | None = None, store: RunStore | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = store or RunStore(settings.db_path)
    app = FastAPI(title="Exact Permutation Test Service", version=__version__)

    @app.exception_handler(DomainError)
    async def domain_handler(_: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": {"code": exc.code.value, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": ErrorCode.INVALID_REQUEST.value,
                    "message": "request failed schema validation",
                    "details": {"issues": jsonable_encoder(exc.errors())},
                }
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_handler(_: Request, exc: Exception) -> JSONResponse:
        log_event(logger, logging.ERROR, "internal_error", error=repr(exc))
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": ErrorCode.INTERNAL_ERROR.value,
                    "message": "unexpected internal error",
                    "details": {},
                }
            },
        )

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "service_version": __version__,
            "kernel_version": KERNEL_VERSION,
            "library_versions": library_versions(),
        }

    @app.post(
        "/v1/permutation-tests",
        response_model=PermutationTestResponse,
        responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    def run_permutation_test(req: PermutationTestRequest) -> PermutationTestResponse:
        run_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        request_payload = req.model_dump()
        input_hash = _input_hash(request_payload)

        total_n = len(req.group_a) + len(req.group_b)
        if total_n > settings.max_group_total:
            raise DomainError(
                ErrorCode.INVALID_PARAMETER,
                f"combined sample size {total_n} exceeds limit {settings.max_group_total}",
            )
        if req.n_permutations is not None and not (
            settings.min_n_permutations <= req.n_permutations <= settings.max_n_permutations
        ):
            raise DomainError(
                ErrorCode.INVALID_PARAMETER,
                f"n_permutations must be within [{settings.min_n_permutations}, "
                f"{settings.max_n_permutations}]",
                details={"n_permutations": req.n_permutations},
            )

        seed = req.seed if req.seed is not None else secrets.randbelow(2**63)
        n_permutations = req.n_permutations or settings.default_n_permutations
        threshold = req.exact_threshold or settings.exact_threshold

        log_event(
            logger, logging.INFO, "run_started",
            run_id=run_id, input_hash=input_hash,
            group_sizes=[len(req.group_a), len(req.group_b)],
            alternative=req.alternative, seed=seed, n_permutations=n_permutations,
            exact_threshold=threshold, max_evaluations=req.max_evaluations,
            kernel_version=KERNEL_VERSION, library_versions=library_versions(),
        )

        def _progress(evaluated: int, planned_total: int) -> None:
            log_event(
                logger, logging.INFO, "run_progress",
                run_id=run_id, evaluated=evaluated, planned_total=planned_total,
            )

        result = permutation_test(
            req.group_a,
            req.group_b,
            alternative=req.alternative,
            n_permutations=n_permutations,
            seed=seed,
            exact_threshold=threshold,
            max_evaluations=req.max_evaluations,
            include_distribution=req.include_distribution,
            progress_cb=_progress,
            chunk_size=settings.chunk_size,
        )

        distribution = result.distribution
        if distribution is not None and len(distribution) > settings.distribution_max_size:
            distribution = None
        response_payload = PermutationTestResponse(
            run_id=run_id,
            status=result.status,
            method=result.method,
            statistic=result.statistic,
            p_value=result.p_value,
            n_permutations=result.n_evaluated,
            n_combinations=result.n_combinations,
            mc_std_error=result.mc_std_error,
            alternative=req.alternative,
            seed=result.seed,
            group_sizes=[len(req.group_a), len(req.group_b)],
            input_hash=input_hash,
            kernel_version=KERNEL_VERSION,
            library_versions=library_versions(),
            created_at=created_at,
            distribution=distribution,
        )
        store.save_run(
            {
                "run_id": run_id,
                "created_at": created_at,
                "status": result.status,
                "input_hash": input_hash,
                "request": request_payload,
                "response": response_payload.model_dump(),
                "kernel_version": KERNEL_VERSION,
                "library_versions": library_versions(),
            }
        )

        if result.status == STATUS_CANCELLED:
            log_event(
                logger, logging.WARNING, "run_cancelled",
                run_id=run_id, n_evaluated=result.n_evaluated,
                n_combinations=result.n_combinations,
            )
            raise DomainError(
                ErrorCode.RUN_CANCELLED,
                "run stopped before completing the permutation sweep; "
                "no valid p-value was produced",
                details={
                    "run_id": run_id,
                    "n_evaluated": result.n_evaluated,
                    "n_combinations": result.n_combinations,
                },
                http_status=409,
            )

        log_event(
            logger, logging.INFO, "run_completed",
            run_id=run_id, input_hash=input_hash, method=result.method,
            p_value=result.p_value, n_evaluated=result.n_evaluated,
            n_combinations=result.n_combinations,
            decision_basis=f"combinations={result.n_combinations} vs exact_threshold={threshold}",
        )
        return response_payload

    @app.get(
        "/v1/permutation-tests/{run_id}",
        response_model=PermutationTestResponse,
        responses={404: {"model": ErrorResponse}},
    )
    def get_permutation_test(run_id: str) -> dict:
        record = store.get_run(run_id)
        if record is None or record["response"] is None:
            raise DomainError(
                ErrorCode.RUN_NOT_FOUND,
                f"no run with id {run_id!r}",
                details={"run_id": run_id},
                http_status=404,
            )
        return record["response"]

    @app.get("/v1/runs")
    def list_runs(limit: int = Query(default=50, ge=1, le=500)) -> dict:
        runs = store.list_runs(limit=limit)
        return {
            "runs": [
                {
                    "run_id": r["run_id"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "input_hash": r["input_hash"],
                    "kernel_version": r["kernel_version"],
                }
                for r in runs
            ]
        }

    return app
