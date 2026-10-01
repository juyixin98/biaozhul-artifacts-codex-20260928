"""FastAPI entry point.

Thin transport layer: parses payloads, assigns/propagates request ids,
delegates to ``path.fit_path``, and maps domain outcomes to HTTP semantics:

  * 200 with ``decision`` in {accepted, undetermined} when a fit ran;
  * 422 with diagnostics when validation rejected the request;
  * request/response logging carries the request id and desensitized state
    only (shapes, counts, gaps) — never raw sample values.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from scipy import sparse

from . import __version__
from .config import Settings, get_settings
from .diagnostics import Decision
from .models import (DiagnosticOut, ErrorResponse, FitRequest, FitResponse,
                     LambdaSolutionOut, PreprocessingInfo, SparseMatrixPayload)
from .path import fit_path
from .validation import InputRejected

logger = logging.getLogger("lasso_path")


def _malformed(request_id: str, shape: str, why: str):
    from . import diagnostics as diag
    return diag.error(request_id, diag.MALFORMED_SPARSE,
                      "sparse payload failed structural checks",
                      declared_shape=shape, reason=why)


def _to_matrix(payload, request_id: str) -> tuple[object, str]:
    if isinstance(payload, SparseMatrixPayload):
        cls = sparse.csc_matrix if payload.format == "csc" else sparse.csr_matrix
        try:
            mat = cls(
                (np.asarray(payload.data, dtype=np.float64),
                 np.asarray(payload.indices, dtype=np.int64),
                 np.asarray(payload.indptr, dtype=np.int64)),
                shape=tuple(payload.shape),
            )
            mat.sum_duplicates()
            mat.check_format(full_check=True)
        except (ValueError, IndexError) as exc:
            raise InputRejected(
                [_malformed(request_id, str(tuple(payload.shape)), type(exc).__name__)])
        return mat, payload.format
    return payload, "dense"


def _diag_out(diags) -> list[DiagnosticOut]:
    return [DiagnosticOut(**d.to_dict()) for d in diags]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = FastAPI(title="lasso-path", version=__version__)

    @app.exception_handler(RequestValidationError)
    async def _schema_error(request: Request, exc: RequestValidationError):
        # Strip 'input'/'ctx' so raw (possibly sensitive) values never leak.
        errors = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")}
                  for e in exc.errors()]
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        logger.warning("request_id=%s schema validation failed: %d error(s)",
                       request_id, len(errors))
        return JSONResponse(status_code=422, content={
            "request_id": request_id,
            "decision": Decision.REJECTED.value,
            "diagnostics": [{
                "request_id": request_id,
                "code": "SCHEMA_VALIDATION",
                "severity": "error",
                "decision": Decision.REJECTED.value,
                "message": "request body failed schema validation",
                "context": {"errors": errors},
            }],
        })

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__}

    @app.post("/v1/lasso/path",
              response_model=FitResponse,
              responses={422: {"model": ErrorResponse}})
    async def fit(request: FitRequest, raw_request: Request):
        request_id = (raw_request.headers.get("x-request-id")
                      or request.request_id
                      or str(uuid.uuid4()))
        try:
            X, matrix_kind = _to_matrix(request.x, request_id)
        except InputRejected as exc:
            logger.warning("request_id=%s rejected: malformed sparse payload", request_id)
            return JSONResponse(status_code=422, content=ErrorResponse(
                request_id=request_id, diagnostics=_diag_out(exc.diagnostics)
            ).model_dump())

        try:
            result = fit_path(
                X, np.asarray(request.y, dtype=np.float64),
                lambdas=request.lambdas,
                n_lambdas=request.n_lambdas,
                lambda_min_ratio=request.lambda_min_ratio,
                fit_intercept=request.fit_intercept,
                tol=request.tol,
                max_iter=request.max_iter,
                zero_variance_policy=request.zero_variance_policy,
                request_id=request_id,
                settings=settings,
            )
        except InputRejected as exc:
            logger.warning("request_id=%s rejected: %s", request_id,
                           exc.diagnostics[0].code)
            return JSONResponse(status_code=422, content=ErrorResponse(
                request_id=request_id, diagnostics=_diag_out(exc.diagnostics)
            ).model_dump())

        logger.info(
            "request_id=%s decision=%s matrix=%s n_lambdas=%d total_iter=%d",
            request_id, result.decision.value, matrix_kind,
            len(result.solutions), sum(s.n_iter for s in result.solutions))

        return FitResponse(
            request_id=request_id,
            decision=result.decision.value,
            preprocessing=PreprocessingInfo(
                centered=request.fit_intercept,
                dropped_columns=result.dropped_columns,
            ),
            solutions=[
                LambdaSolutionOut(
                    **{"lambda": s.lambda_},
                    intercept=s.intercept,
                    n_nonzero=int(s.support.size),
                    support=[int(i) for i in s.support],
                    values=[float(v) for v in s.coef[s.support]],
                    converged=s.converged,
                    n_iter=s.n_iter,
                    duality_gap=s.duality_gap,
                    kkt_violation=s.kkt_violation,
                )
                for s in result.solutions
            ],
            diagnostics=_diag_out(result.diagnostics),
        )

    return app


app = create_app()
