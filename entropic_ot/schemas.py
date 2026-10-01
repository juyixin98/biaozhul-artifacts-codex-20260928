"""Pydantic request and response contracts."""

from typing import Literal

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_mass: list[float] = Field(min_length=1)
    target_mass: list[float] = Field(min_length=1)
    cost: list[list[float]]
    epsilon: float = Field(gt=0)
    max_iterations: int = Field(default=10_000, ge=1, le=1_000_000)
    atol: float = Field(default=1e-10, ge=0)
    rtol: float = Field(default=1e-8, ge=0)

    @field_validator("epsilon", "atol", "rtol")
    @classmethod
    def reject_non_finite_float(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return value


class IterationEventResponse(BaseModel):
    iteration: int
    row_marginal_error: float
    column_marginal_error: float
    combined_error_ratio: float
    finite_log_plan: bool


class DiagnosticsResponse(BaseModel):
    run_id: str
    status: Literal["converged", "failed"]
    stopped_reason: str
    feasible: bool
    converged: bool
    iterations: int
    max_iterations: int
    active_shape: list[int]
    row_marginal_error: float
    column_marginal_error: float
    combined_error_ratio: float
    row_tolerance: float
    column_tolerance: float
    events: list[IterationEventResponse]


class TransportResponse(BaseModel):
    run_id: str
    plan: list[list[float]]
    transport_cost: float
    regularized_objective: float
    entropy_term: float
    kl_to_uniform: float
    diagnostics: DiagnosticsResponse


class ErrorResponse(BaseModel):
    error: Literal[
        "input_error",
        "state_conflict",
        "resource_exhausted",
        "computation_failed",
    ]
    message: str
    run_id: str | None = None
    details: dict
    diagnostics: dict | None = None
