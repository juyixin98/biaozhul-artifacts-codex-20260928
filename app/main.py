import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.config import SERVICE_NAME
from app.errors import QPError
from app.models import FeasibilityRequest, SolveRequest
from app.observability import configure_logging, log_event, new_request_id, request_id_var
from app.service import find_feasibility, runtime_metadata, solve_qp


logger = configure_logging()

app = FastAPI(
    title="Constrained Quadratic Programming Active-Set Service",
    version=__version__,
    description=(
        "Dense positive-definite QP with linear equality and inequality constraints. "
        "Phase 1 feasibility and Phase 2 primal active-set iteration are separate."
    ),
)


@app.exception_handler(QPError)
async def qp_error_handler(request: Request, exc: QPError) -> JSONResponse:
    request_id = getattr(exc, "request_id", None) or request_id_var.get() or new_request_id()
    token = request_id_var.set(request_id)
    log_event(logger, logging.ERROR, "QP request failed", code=exc.code, path=request.url.path)
    request_id_var.reset(token)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "failed",
            "failure_category": exc.code,
            "failure_reasons": [exc.message],
            "uncertainties": [],
            "request_id": request_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = new_request_id()
    return JSONResponse(
        status_code=422,
        content={
            "request_id": request_id,
            "status": "failed",
            "failure_category": "request_validation_error",
            "failure_reasons": [
                ".".join(str(part) for part in error["loc"]) + f": {error['msg']}"
                for error in exc.errors()
            ],
            "uncertainties": [],
        },
    )


@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"ok": True, "service": SERVICE_NAME, "version": __version__, "runtime": runtime_metadata()}


@app.post("/api/v1/qp/solve", tags=["qp"])
async def solve(request: SolveRequest) -> dict:
    return solve_qp(request)


@app.post("/api/v1/qp/feasibility", tags=["qp"])
async def feasibility(request: FeasibilityRequest) -> dict:
    return find_feasibility(request)


@app.get("/api/v1/qp/scope", tags=["qp"])
async def scope() -> dict:
    return {
        "supported": [
            "dense symmetric positive-definite Hessian",
            "linear <=, >=, and equality constraints",
            "independent feasibility phase via SciPy HiGHS",
            "primal active-set main iteration",
            "SVD/rank-revealing QR rank handling",
            "final primal, dual, stationarity, and complementarity validation",
        ],
        "not_supported": [
            "non-convex or semidefinite quadratic programs",
            "sparse or very large-scale production data",
            "integer variables",
            "external data sources or cloud accounts",
        ],
    }
