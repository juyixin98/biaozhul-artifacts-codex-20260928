"""Pydantic request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ParameterSpec(BaseModel):
    initial: float
    lower: float = Field(..., allow_inf_nan=False)
    upper: float = Field(..., allow_inf_nan=False)
    scale: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ParameterSpec":
        if self.lower >= self.upper:
            raise ValueError("lower must be strictly less than upper")
        if not self.lower <= self.initial <= self.upper:
            raise ValueError("initial must satisfy lower <= initial <= upper")
        if self.scale is not None and self.scale <= 0.0:
            raise ValueError("scale must be positive")
        return self


class SolveOptionsSchema(BaseModel):
    max_iterations: int = Field(default=50, ge=0, le=1000)
    tolerance: float = Field(default=1e-3, gt=0.0, lt=1.0)
    initial_trust_radius: float = Field(default=0.3, gt=0.0)
    max_trust_radius: float = Field(default=10.0, gt=0.0)
    min_trust_radius: float = Field(default=1e-12, gt=0.0)
    derivative_check: bool = True
    derivative_tolerance: float = Field(default=5e-6, gt=0.0)
    rank_tolerance: float = Field(default=1e-10, gt=0.0)


class SolveRequest(BaseModel):
    expression: str = Field(..., min_length=1, max_length=1000)
    parameters: dict[str, ParameterSpec]
    x: list[float]
    y: list[float]
    options: SolveOptionsSchema = Field(default_factory=SolveOptionsSchema)
    run_id: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_data(self) -> "SolveRequest":
        if not self.parameters:
            raise ValueError("at least one parameter is required")
        if len(self.x) == 0:
            raise ValueError("x must not be empty")
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")
        if len(self.x) < len(self.parameters):
            raise ValueError("observations must be at least as numerous as parameters")
        if any(not value == value or value in (float("inf"), float("-inf")) for value in self.x):
            raise ValueError("x contains NaN or infinity")
        if any(not value == value or value in (float("inf"), float("-inf")) for value in self.y):
            raise ValueError("y contains NaN or infinity")
        return self


class HealthResponse(BaseModel):
    service: str
    status: Literal["ok"]
    version: str


class SolveResponse(BaseModel):
    success: bool
    status: str
    reason: str
    parameters: dict[str, float]
    residuals: list[float]
    cost: float
    iterations: int
    accepted_iterations: int
    stationary_type: str
    boundary_parameters: list[str]
    diagnostics: dict
    run_id: str
