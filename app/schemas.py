"""FastAPI request and response schemas."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import SolverConfig


class ParameterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    initial: float
    lower: float = -math.inf
    upper: float = math.inf
    scale: float = Field(default=1.0, gt=0.0)


class SolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(min_length=1, max_length=2000)
    x: list[float]
    y: list[float]
    parameters: list[ParameterSpec] = Field(min_length=1)
    config: SolverConfig = Field(default_factory=SolverConfig)

    @field_validator("x", "y")
    @classmethod
    def finite_data(cls, values: list[float]) -> list[float]:
        if not values:
            raise ValueError("data must not be empty")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("x and y must be finite")
        if len(values) > 100000:
            raise ValueError("at most 100000 data points are accepted")
        return values

    @field_validator("parameters")
    @classmethod
    def valid_parameters(cls, parameters: list[ParameterSpec]) -> list[ParameterSpec]:
        names = [parameter.name for parameter in parameters]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        for parameter in parameters:
            if parameter.lower >= parameter.upper:
                raise ValueError(f"parameter {parameter.name} requires lower < upper")
            if not (parameter.lower <= parameter.initial <= parameter.upper):
                raise ValueError(f"initial value for {parameter.name} must be within bounds")
        return parameters


class ErrorResponse(BaseModel):
    run_id: str
    status: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    log_tail: list[dict[str, Any]] = Field(default_factory=list)
