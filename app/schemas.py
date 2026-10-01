from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.diagnostics import FitStatus, Identifiability, IterationRecord


class BoundsSpec(BaseModel):
    lower: list[float | None]
    upper: list[float | None]


class FitOptions(BaseModel):
    max_iterations: int = Field(default=100, ge=1, le=1000)
    initial_radius: float = Field(default=1.0, gt=0.0, le=1.0e6)
    min_radius: float = Field(default=1.0e-12, gt=0.0)
    gtol: float = Field(default=1.0e-8, gt=0.0)
    xtol: float = Field(default=1.0e-10, gt=0.0)
    ftol: float = Field(default=1.0e-12, gt=0.0)
    max_invalid_trials: int = Field(default=20, ge=0, le=200)
    check_derivatives: bool = True
    derivative_tolerance: float = Field(default=5e-5, gt=0.0, lt=1.0)
    finite_difference_step: float = Field(default=1e-7, gt=0.0, lt=0.1)


class FitRequest(BaseModel):
    expression: str = Field(min_length=1, max_length=1000)
    parameter_names: list[str] = Field(min_length=1, max_length=50)
    x: list[float]
    y: list[float]
    weights: list[float] | None = None
    initial: list[float]
    bounds: BoundsSpec
    scales: list[float] | None = None
    variable_name: str = Field(default="x", min_length=1, max_length=32)
    options: FitOptions = Field(default_factory=FitOptions)

    @field_validator("parameter_names")
    @classmethod
    def unique_names(cls, names: list[str]) -> list[str]:
        if not all(name.isidentifier() for name in names):
            raise ValueError("parameter names must be valid Python identifiers")
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        return names

    @field_validator("variable_name")
    @classmethod
    def valid_identifier(cls, value: str) -> str:
        if not value.isidentifier():
            raise ValueError("variable_name must be a valid Python identifier")
        return value


class IterationResponse(BaseModel):
    iteration: int
    accepted: bool
    cost: float
    actual_reduction: float | None
    predicted_reduction: float | None
    reduction_ratio: float | None
    radius: float
    step_norm_scaled: float
    projected_gradient_norm: float
    active_lower: list[str]
    active_upper: list[str]
    bound_hit: list[str]
    judgment: str

    @classmethod
    def from_core(cls, item: IterationRecord) -> "IterationResponse":
        return cls(**asdict(item))


class IdentifiabilityResponse(BaseModel):
    rank: int
    parameter_count: int
    identifiable: bool
    singular_values: list[float]
    condition_number: float | None
    parameter_correlation: list[list[float | None]]

    @classmethod
    def from_core(cls, item: Identifiability) -> "IdentifiabilityResponse":
        return cls(**asdict(item))


class DerivativeCheckResponse(BaseModel):
    passed: bool
    max_relative_error: float
    worst_parameter: str | None
    worst_observation: int | None
    relative_step: float


class FitResponse(BaseModel):
    run_id: str
    input_fingerprint: str
    success: bool
    status: FitStatus
    message: str
    reason: str
    parameters: dict[str, float]
    residuals: list[float]
    cost: float
    projected_gradient_norm: float
    stationarity_type: Literal["interior", "boundary", "mixed"]
    active_lower: dict[str, bool]
    active_upper: dict[str, bool]
    scales: dict[str, float]
    nfev: int
    njev: int
    invalid_evaluations: int
    final_radius: float
    iterations: list[IterationResponse]
    identifiability: IdentifiabilityResponse
    derivative_check: DerivativeCheckResponse | None


class ErrorResponse(BaseModel):
    run_id: str
    success: Literal[False] = False
    status: FitStatus
    message: str
    reason: str
