"""Pydantic request and response contracts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: list[float] = Field(
        ...,
        min_length=1,
        description="Non-negative source masses.",
    )
    b: list[float] = Field(
        ...,
        min_length=1,
        description="Non-negative target masses with the same total as a.",
    )
    cost: list[list[float]] = Field(
        ...,
        min_length=1,
        description="Finite transport cost matrix, shape len(a) by len(b).",
    )
    epsilon: float = Field(..., gt=0.0, allow_inf_nan=False)
    max_iterations: int = Field(default=100_000, gt=0, le=1_000_000)
    tolerance: float = Field(default=1e-10, gt=0.0, allow_inf_nan=False)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("a", "b")
    @classmethod
    def validate_marginals(cls, value: list[float]) -> list[float]:
        if not all(math.isfinite(x) and x >= 0.0 for x in value):
            raise ValueError("all masses must be finite and non-negative")
        return value

    @field_validator("cost")
    @classmethod
    def validate_cost(cls, value: list[list[float]]) -> list[list[float]]:
        if any(not isinstance(row, list) for row in value):
            raise ValueError("cost must be a matrix")
        if any(not all(math.isfinite(x) for x in row) for row in value):
            raise ValueError("all costs must be finite")
        widths = {len(row) for row in value}
        if len(widths) != 1:
            raise ValueError("all cost rows must have the same length")
        return value


class DiagnosticsResponse(BaseModel):
    tolerance: float
    total_mass: float
    active_rows: list[int]
    active_columns: list[int]
    log_row_error_at_stop: float
    log_column_error_at_stop: float
    plan_mass: float
    feasibility_reason: str


class SolveResponse(BaseModel):
    run_id: str
    plan: list[list[float]]
    original_cost: float
    regularized_cost: float
    entropy_penalty: float
    epsilon: float
    iterations: int
    max_iterations: int
    converged: bool
    feasible: bool
    stop_reason: str
    row_error: float
    column_error: float
    row_marginals: list[float]
    column_marginals: list[float]
    diagnostics: DiagnosticsResponse
    replay: dict[str, Any]


class ErrorBody(BaseModel):
    category: str
    code: str
    message: str
    details: dict[str, Any]
    run_id: str | None


class ErrorResponse(BaseModel):
    error: ErrorBody


def serialize_result(
    run_id: str,
    result: Any,
    *,
    request_hash: str | None,
    replayed: bool = False,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "plan": np.asarray(result.plan, dtype=np.float64).tolist(),
        "original_cost": result.original_cost,
        "regularized_cost": result.regularized_cost,
        "entropy_penalty": result.entropy_penalty,
        "epsilon": result.epsilon,
        "iterations": result.iterations,
        "max_iterations": result.max_iterations,
        "converged": result.converged,
        "feasible": result.feasible,
        "stop_reason": result.stop_reason,
        "row_error": result.row_error,
        "column_error": result.column_error,
        "row_marginals": np.asarray(result.row_marginals, dtype=np.float64).tolist(),
        "column_marginals": np.asarray(result.column_marginals, dtype=np.float64).tolist(),
        "diagnostics": result.diagnostics,
        "replay": {
            "request_hash": request_hash,
            "log_file": f"{run_id}.jsonl",
            "replayed": replayed,
        },
    }
