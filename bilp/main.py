"""FastAPI service entrypoint.

Endpoints:
  GET  /healthz                            liveness
  GET  /v1/config                          active solver configuration
  POST /v1/solve                           solve a bounded MILP
  GET  /v1/runs/{run_id}                   stored run (problem + result)
  POST /v1/runs/{run_id}/replay            deterministic re-solve + comparison
  GET  /v1/runs/{run_id}/nodes/{node_id}   restore a node (branch constraints + bounds)
"""

from __future__ import annotations

import dataclasses

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .config import SolverConfig, load_config
from .errors import (
    BilpError,
    ErrorCategory,
    InputValidationError,
    ResourceExhaustedError,
    RunNotFoundError,
)
from .model import parse_problem
from .runs import RunStore, canonical_fingerprint, make_run_id, new_run_record
from .schemas import ProblemSpec, SolveOptions, SolveRequest, SolveResponse
from .solver.branch_and_bound import BranchAndBoundSolver


def effective_config(base: SolverConfig, options: SolveOptions) -> SolverConfig:
    """Merge per-request options over the server config, enforcing hard caps."""
    overrides: dict[str, object] = {}
    if options.max_nodes is not None:
        if options.max_nodes < 1:
            raise InputValidationError(
                "options.max_nodes must be >= 1", {"max_nodes": options.max_nodes}
            )
        if options.max_nodes > base.hard_max_nodes:
            raise ResourceExhaustedError(
                f"requested node budget {options.max_nodes} exceeds server hard cap "
                f"{base.hard_max_nodes}",
                {"requested": options.max_nodes, "hard_cap": base.hard_max_nodes},
            )
        overrides["max_nodes"] = options.max_nodes
    if options.time_limit_s is not None:
        if options.time_limit_s <= 0:
            raise InputValidationError(
                "options.time_limit_s must be > 0", {"time_limit_s": options.time_limit_s}
            )
        if options.time_limit_s > base.hard_max_time_s:
            raise ResourceExhaustedError(
                f"requested time limit {options.time_limit_s}s exceeds server hard cap "
                f"{base.hard_max_time_s}s",
                {"requested": options.time_limit_s, "hard_cap": base.hard_max_time_s},
            )
        overrides["time_limit_s"] = options.time_limit_s
    if options.int_tol is not None:
        if not (0.0 < options.int_tol < 0.5):
            raise InputValidationError(
                "options.int_tol must be in (0, 0.5)", {"int_tol": options.int_tol}
            )
        overrides["int_tol"] = options.int_tol
    return dataclasses.replace(base, **overrides)


def create_app(config: SolverConfig | None = None, store: RunStore | None = None) -> FastAPI:
    config = config or load_config()
    store = store or RunStore(config.runs_log_path)
    app = FastAPI(title="bilp-service", version=__version__)

    @app.exception_handler(BilpError)
    async def bilp_error_handler(_: Request, exc: BilpError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={
            "error": {
                "category": ErrorCategory.INPUT_ERROR.value,
                "message": "request validation failed",
                "details": {"errors": jsonable_encoder(exc.errors())},
            }
        })

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={
            "error": {
                "category": ErrorCategory.COMPUTATION_FAILURE.value,
                "message": f"unexpected internal error: {type(exc).__name__}",
                "details": {},
            }
        })

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/config")
    def get_config() -> dict:
        return dataclasses.asdict(config)

    @app.post("/v1/solve", response_model=SolveResponse)
    def solve(req: SolveRequest) -> dict:
        problem_dict = req.problem.model_dump()
        options_dict = req.options.model_dump()
        fingerprint = canonical_fingerprint(problem_dict, options_dict)

        if req.client_request_id is not None:
            existing = store.check_idempotency(req.client_request_id, fingerprint)
            if existing is not None:
                body = dict(existing.result)
                if not req.options.include_evidence:
                    body["evidence"] = None
                body["idempotent_replay"] = True
                return body

        cfg = effective_config(config, req.options)
        problem = parse_problem(req.problem, cfg)

        run_id = make_run_id(problem_dict, options_dict, __version__)
        cached = store.get(run_id)
        if cached is not None:
            if req.client_request_id is not None:
                store.register_idempotency(req.client_request_id, fingerprint, run_id)
            body = dict(cached.result)
            if not req.options.include_evidence:
                body["evidence"] = None
            body["idempotent_replay"] = True
            return body

        solver = BranchAndBoundSolver(problem, cfg, run_id=run_id)
        result = solver.solve()
        result_dict = result.to_dict(include_evidence=True)
        store.put(new_run_record(run_id, problem_dict, options_dict, result_dict, result.archive))
        if req.client_request_id is not None:
            store.register_idempotency(req.client_request_id, fingerprint, run_id)

        body = dict(result_dict)
        if not req.options.include_evidence:
            body["evidence"] = None
        return body

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        record = store.get(run_id)
        if record is None:
            raise RunNotFoundError(f"unknown run id {run_id!r}", {"run_id": run_id})
        return record.summary()

    @app.post("/v1/runs/{run_id}/replay")
    def replay_run(run_id: str) -> dict:
        record = store.get(run_id)
        if record is None:
            raise RunNotFoundError(f"unknown run id {run_id!r}", {"run_id": run_id})
        spec = ProblemSpec(**record.problem)
        options = SolveOptions(**record.options)
        cfg = effective_config(config, options)
        problem = parse_problem(spec, cfg)
        solver = BranchAndBoundSolver(problem, cfg, run_id=record.run_id)
        result = solver.solve()
        original = record.result
        matches = (
            result.status.value == original["status"]
            and result.objective == original["objective"]
            and result.diagnostics["nodes_processed"] == original["diagnostics"]["nodes_processed"]
        )
        return {
            "run_id": record.run_id,
            "matches_original": matches,
            "original": {
                "status": original["status"],
                "objective": original["objective"],
                "nodes_processed": original["diagnostics"]["nodes_processed"],
            },
            "replayed": result.to_dict(include_evidence=False),
        }

    @app.get("/v1/runs/{run_id}/nodes/{node_id}")
    def get_node(run_id: str, node_id: int) -> dict:
        record = store.get(run_id)
        if record is None:
            raise RunNotFoundError(f"unknown run id {run_id!r}", {"run_id": run_id})
        if record.archive is None:
            raise RunNotFoundError(
                "node archive for this run is not retained in memory "
                "(run was reloaded from the log); replay the run to rebuild it",
                {"run_id": run_id, "node_id": node_id},
            )
        return record.archive.restore(node_id)

    return app


app = create_app()
