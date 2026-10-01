"""Pydantic HTTP request and response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DecompositionRequest(BaseModel):
    values: list[float] = Field(
        ...,
        min_length=1,
        description="Complete observed series; missing values must be explicit "
        "and are not accepted by this backend.",
    )
    run_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Optional caller-supplied idempotency key.",
    )


class Components(BaseModel):
    trend: list[float]
    seasonal: list[float]
    residual: list[float]


class DecompositionResponse(BaseModel):
    run_id: str
    run_number: int
    status: str
    schema_version: str
    kernel_version: str
    config: dict[str, Any]
    observed: list[float]
    components: Components
    robust_weights: list[float]
    diagnostics: dict[str, Any]
    iteration_history: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    error: dict[str, Any]
