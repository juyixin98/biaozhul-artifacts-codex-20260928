"""Pydantic schemas for the HTTP service.

These are transport shapes only; all domain decisions live in
``validation`` / ``path`` and surface here as diagnostics.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SparseMatrixPayload(BaseModel):
    """CSR/CSC matrix in explicit index form (never densified server-side)."""

    format: Literal["csc", "csr"]
    data: list[float]
    indices: list[int]
    indptr: list[int]
    shape: tuple[int, int]


class FitRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    x: list[list[float]] | SparseMatrixPayload = Field(alias="X")
    y: list[float]
    lambdas: list[float] | None = None
    n_lambdas: int | None = Field(default=None, ge=1)
    lambda_min_ratio: float | None = Field(default=None, gt=0.0, lt=1.0)
    fit_intercept: bool = True
    tol: float | None = Field(default=None, gt=0.0)
    max_iter: int | None = Field(default=None, ge=1)
    zero_variance_policy: Literal["reject", "drop"] = "reject"
    request_id: str | None = None


class DiagnosticOut(BaseModel):
    request_id: str
    code: str
    severity: str
    decision: str
    message: str
    context: dict[str, Any]


class LambdaSolutionOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    lambda_: float = Field(alias="lambda")
    intercept: float
    n_nonzero: int
    support: list[int]
    values: list[float]  # coefficients on the support, original scale
    converged: bool
    n_iter: int
    duality_gap: float
    kkt_violation: float


class PreprocessingInfo(BaseModel):
    centered: bool
    scaling: str = "population_std"
    intercept_penalized: bool = False
    dropped_columns: list[int]


class FitResponse(BaseModel):
    request_id: str
    decision: str
    preprocessing: PreprocessingInfo
    solutions: list[LambdaSolutionOut]
    diagnostics: list[DiagnosticOut]


class ErrorResponse(BaseModel):
    request_id: str
    decision: str = "rejected"
    diagnostics: list[DiagnosticOut]
