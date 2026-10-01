"""Application service that coordinates samples, estimators, persistence, and logging."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.covariance.estimators import ESTIMATORS
from app.covariance.sample import MethodName, prepare_sample
from app.errors import CovarianceError, FailureRecord
from app.logging_config import log_event
from app.models import EstimationRequest
from app.storage import RequestJournal
from app.versioning import envelope


def resolve_request_id(request: EstimationRequest, header_request_id: str | None) -> str:
    candidates = [value for value in [header_request_id, request.request_id] if value]
    if not candidates:
        return f"cov-{uuid.uuid4().hex}"
    if len(set(candidates)) > 1:
        raise CovarianceError(
            "Header request_id and body request_id must match when both are supplied.",
            stage="request_identity",
            details={"header": header_request_id, "body": request.request_id},
        )
    return candidates[0]


def _uncertainties(method_results: list[tuple[str, dict[str, Any] | None]]) -> list[dict[str, Any]]:
    uncertainties: list[dict[str, Any]] = []
    for method, result in method_results:
        if result is None:
            continue
        diagnostics = result["matrix_diagnostics"]
        if not diagnostics["invertible"]:
            uncertainties.append(
                {
                    "scope": method,
                    "conclusion": (
                        f"Rank {diagnostics['rank']} of {diagnostics['n_features']} means the matrix "
                        "is not invertible; downstream inverse covariance is unavailable."
                    ),
                }
            )
        for warning in result.get("warnings", []):
            if "asymptotic" in warning or "cautiously" in warning:
                uncertainties.append({"scope": method, "conclusion": warning})
    return uncertainties


def estimate(
    request: EstimationRequest,
    *,
    journal: RequestJournal,
    logger: logging.Logger,
    request_id: str,
) -> dict[str, Any]:
    payload = request.model_dump(mode="json")
    sample_report: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    method_results: list[tuple[str, dict[str, Any] | None]] = []
    n_samples: int | None = None
    n_features: int | None = None

    log_event(
        logger,
        logging.INFO,
        "request_received",
        "Covariance estimation request received.",
        request_id=request_id,
        details={"methods": request.methods, "n_rows": len(request.rows)},
    )
    try:
        prepared = prepare_sample(request.rows, request.feature_names)
        sample_report = prepared.as_report()
        n_samples = prepared.n_samples
        n_features = prepared.n_features
        log_event(
            logger,
            logging.INFO,
            "sample_prepared",
            "Sample contract and fixed preprocessing completed.",
            request_id=request_id,
            details={
                "n_samples": n_samples,
                "n_features": n_features,
                "missing_cells": prepared.missing_cells,
                "location": "app/covariance/sample.py",
            },
        )

        for requested_method in request.methods:
            method = MethodName(requested_method)
            try:
                log_event(
                    logger,
                    logging.INFO,
                    "method_started",
                    f"Starting {method} estimation.",
                    request_id=request_id,
                    details={"method": str(method), "location": "app/covariance/estimators.py"},
                )
                result = ESTIMATORS[method](prepared).to_dict()
                results.append(result)
                method_results.append((str(method), result))
                log_event(
                    logger,
                    logging.INFO,
                    "method_completed",
                    f"Completed {method} estimation.",
                    request_id=request_id,
                    details={
                        "method": str(method),
                        "rank": result["matrix_diagnostics"]["rank"],
                        "invertible": result["matrix_diagnostics"]["invertible"],
                    },
                )
            except CovarianceError as error:
                failure = FailureRecord.from_error(error)
                failures.append(failure)
                method_results.append((str(method), None))
                log_event(
                    logger,
                    logging.ERROR,
                    "method_failed",
                    f"Method {method} failed: {error.message}",
                    request_id=request_id,
                    details=failure.to_dict(),
                )

        status = "success" if not failures else ("partial_success" if results else "failed")
        response = envelope(
            request_id=request_id,
            status=status,
            sample_report=sample_report,
            results=results,
            failures=failures,
            uncertainties=_uncertainties(method_results),
        )
        journal.record(
            request_id=request_id,
            created_at=response["created_at"],
            status=status,
            methods=list(map(str, request.methods)),
            n_samples=n_samples,
            n_features=n_features,
            payload=payload,
            response=response,
            failure=None,
        )
        log_event(
            logger,
            logging.INFO,
            "request_completed",
            "Covariance estimation request completed.",
            request_id=request_id,
            details={"status": status, "n_results": len(results), "n_failures": len(failures)},
        )
        return response
    except CovarianceError as error:
        failure = FailureRecord.from_error(error)
        response = envelope(
            request_id=request_id,
            status="failed",
            sample_report=sample_report,
            results=results,
            failures=[failure],
            uncertainties=[
                {
                    "scope": "sample" if sample_report is None else "request",
                    "conclusion": error.message,
                }
            ],
        )
        journal.record(
            request_id=request_id,
            created_at=response["created_at"],
            status="failed",
            methods=list(map(str, request.methods)),
            n_samples=n_samples,
            n_features=n_features,
            payload=payload,
            response=response,
            failure=failure.to_dict(),
        )
        log_event(
            logger,
            logging.ERROR,
            "request_failed",
            f"Covariance request failed: {error.message}",
            request_id=request_id,
            details=failure.to_dict(),
        )
        return response
