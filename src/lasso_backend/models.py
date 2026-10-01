"""Pydantic API contracts for the Lasso path service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class LassoPathRequest(BaseModel):
    x: dict[str, Any] | list[list[float]]
    y: list[float]
    lambdas: list[float] | None = Field(default=None, min_length=1)
    n_alphas: int = Field(default=100, ge=2, le=500)
    lambda_min_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    lambda_order: Literal["descending", "ascending"] = "descending"
    warm_start: bool = True
    max_iter: int = Field(default=10_000, ge=1, le=100_000)
    tol: float = Field(default=1e-8, gt=0.0, le=1.0)
    zero_variance_policy: Literal["reject", "zero_fill"] = "reject"
    variance_tolerance: float = Field(default=1e-12, ge=0.0)

    @field_validator("lambdas")
    @classmethod
    def validate_lambdas(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and any(
            lambda_value <= 0.0 for lambda_value in value
        ):
            raise ValueError("every lambda must be strictly positive")
        return value


class LambdaPointResponse(BaseModel):
    lambda_value: float
    coefficients: list[float]
    intercept: float
    support: list[int]
    iterations: int
    converged: bool
    status: str
    reason: str
    kkt_violation: float
    duality_gap: float
    primal_objective: float


class LassoPathResponse(BaseModel):
    request_id: str
    accepted: bool
    decision: Literal["accepted", "undetermined"]
    reason: str
    lambdas: list[float]
    points: list[LambdaPointResponse]
    diagnostics: dict
