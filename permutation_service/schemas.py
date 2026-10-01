"""Pydantic HTTP request and response contracts."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(
        ..., min_length=2, description="Pooled observations in sample-identity order."
    )
    group1_size: int = Field(..., ge=1)
    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    method: Literal["auto", "exact", "monte_carlo"] = "auto"
    monte_carlo_replications: int | None = Field(default=None, ge=1)
    exact_max_combinations: int | None = Field(default=None, ge=0)
    random_seed: int | None = Field(default=None, ge=0)
    return_null_distribution: bool = False

    @field_validator("values")
    @classmethod
    def values_are_finite(cls, values: list[float]) -> list[float]:
        for index, value in enumerate(values):
            if not math.isfinite(value):
                raise ValueError(f"value at index {index} must be finite")
        return values


class ErrorBody(BaseModel):
    code: str
    message: str
    category: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


class RunEventResponse(BaseModel):
    seq: int
    run_id: str
    created_at: str
    phase: str
    completed: int | None
    total: int | None
    batch_size: int | None
    message: str


class RunResponse(BaseModel):
    run_id: str
    status: Literal["pending", "running", "completed", "failed", "canceled"]
    created_at: str
    updated_at: str
    input: dict[str, Any]
    input_hash: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancel_requested: bool


class CancelResponse(BaseModel):
    run_id: str
    status: str
    cancel_requested: bool
    message: str


class VersionResponse(BaseModel):
    service: str
    python: str
    numpy: str
    scipy: str
    fastapi: str
    pydantic: str
    sqlite: str

