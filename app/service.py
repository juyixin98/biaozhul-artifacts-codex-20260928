from __future__ import annotations

import uuid
from time import perf_counter

import numpy as np

from app.config import settings
from app.models import CovarianceRequest, CovarianceResponse, ErrorDetail
from app.sample_contract import SampleContractError, prepare_sample
from app.statistical_kernel import (
    diagnose_matrix,
    ledoit_wolf_shrinkage,
    robust_tyler_shrinkage,
)
from app.versioning import ESTIMATOR_VERSION, SCHEMA_VERSION, SAMPLE_CONTRACT_VERSION


def _failure(code_message: str) -> ErrorDetail:
    code, _, message = code_message.partition(":")
    return ErrorDetail(code=code.strip(), message=message.strip() or code_message)


def _uncertainty(code_message: str) -> ErrorDetail:
    return _failure(code_message)


def _step(name: str, started: float, **details: object) -> dict[str, object]:
    event = {"step": name, "elapsed_ms": round((perf_counter() - started) * 1000.0, 4)}
    event.update(details)
    return event


def estimate(
    request: CovarianceRequest,
    *,
    request_id: str | None = None,
) -> tuple[
    CovarianceResponse, str, list[dict[str, object]], list[dict[str, object]], float
]:
    total_started = perf_counter()
    request_id = request_id or str(uuid.uuid4())
    n = len(request.data)
    p = len(request.data[0])
    if p > settings.max_dimension or n > settings.max_samples:
        raise SampleContractError(
            f"MATRIX_TOO_LARGE: limits are {settings.max_samples} rows and {settings.max_dimension} columns"
        )

    steps: list[dict[str, object]] = []
    preprocess_started = perf_counter()
    center_method = "mean" if request.method == "ledoit_wolf" else "median"
    prepared = prepare_sample(
        request.data,
        assume_centered=request.assume_centered,
        center_method=center_method,
        supplied_scales=request.column_scales,
    )
    steps.append(
        _step(
            "sample.prepare_fixed_contract",
            preprocess_started,
            sample_contract_version=SAMPLE_CONTRACT_VERSION,
            center="zero" if request.assume_centered else center_method,
            missing_values=prepared.missing_count,
            complete_rows=prepared.complete_rows,
            scale_policy="supplied" if request.column_scales else "median_mad_1.4826",
        )
    )

    if request.method == "ledoit_wolf":
        result = ledoit_wolf_shrinkage(prepared.standardized)
    else:
        result = robust_tyler_shrinkage(
            prepared.standardized,
            shrinkage=request.shrinkage,
            max_iterations=request.max_iterations,
            tolerance=request.tolerance,
        )

    rescale_started = perf_counter()
    scale_diag = np.diag(prepared.scales)
    covariance = scale_diag @ result.covariance @ scale_diag
    diagnostics = diagnose_matrix(covariance)
    steps.extend(result.steps)
    steps.append(
        _step(
            "covariance.restore_original_units",
            rescale_started,
            min_eigenvalue=diagnostics.min_eigenvalue,
            max_eigenvalue=diagnostics.max_eigenvalue,
            rank=diagnostics.rank,
            invertible=diagnostics.invertible,
        )
    )

    imputed_values: list[list[float | None]] = []
    for i in range(n):
        row: list[float | None] = []
        for j in range(p):
            row.append(float(prepared.imputed[i, j]) if prepared.mask[i, j] else None)
        imputed_values.append(row)

    failures = [_failure(item).model_dump() for item in result.failures]
    failures.extend(
        _failure(item).model_dump() for item in diagnostics.validation_failures
    )
    uncertainties = [
        _uncertainty(item).model_dump() for item in result.uncertainties
    ]
    if prepared.missing_count:
        uncertainties.append(
            ErrorDetail(
                code="MISSING_IMPUTED",
                message=(
                    f"{prepared.missing_count} entries were median-imputed; "
                    "uncertainty from missingness is not included in covariance"
                ),
            ).model_dump()
        )

    status = "ok" if not failures and not uncertainties else "uncertain"
    if not diagnostics.positive_semidefinite or not diagnostics.symmetric:
        status = "error"

    response = CovarianceResponse(
        schema_version=SCHEMA_VERSION,
        estimator_version=ESTIMATOR_VERSION,
        request_id=request_id,
        status=status,
        method=request.method,
        n_samples=n,
        n_dimensions=p,
        location=[float(x) for x in prepared.center],
        covariance=[[float(x) for x in row] for row in covariance],
        eigenvalues=[float(x) for x in diagnostics.eigenvalues],
        condition_number=diagnostics.condition_number,
        rank=diagnostics.rank,
        invertible=diagnostics.invertible,
        shrinkage=float(result.shrinkage),
        outlier_weights=[float(x) for x in result.weights],
        missing_mask=[[bool(x) for x in row] for row in prepared.mask],
        imputed_values=imputed_values,
        column_scales=[float(x) for x in prepared.scales],
        assumptions=result.assumptions,
        steps=steps,
        failures=[ErrorDetail(**item) for item in failures],
        uncertainties=[ErrorDetail(**item) for item in uncertainties],
    )
    duration_ms = round((perf_counter() - total_started) * 1000.0, 4)
    steps.append(
        _step("response.complete", total_started, status=status, duration_ms=duration_ms)
    )
    return response, status, failures, uncertainties, duration_ms
