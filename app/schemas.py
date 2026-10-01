"""HTTP request and response contracts."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import FIXED_CONFIG


class DecomposeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    values: list[float] = Field(..., min_length=1)

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[float]) -> list[float]:
        for index, value in enumerate(values):
            if isinstance(value, bool):
                raise ValueError("boolean values are not numeric observations")
            if not math.isfinite(value):
                raise ValueError(
                    f"value at index {index} is NaN or infinite; missing values are rejected"
                )
        return values


class DecomposeResponse(BaseModel):
    run_id: str
    trend: list[float]
    seasonal: list[float]
    residual: list[float]
    weights: list[float]
    diagnostics: dict[str, Any]
    trace: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    error: str
    category: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None


class VersionResponse(BaseModel):
    spec_version: str
    model_version: str
    config: dict[str, Any]


FIXED_CONTRACT_CONFIG = FIXED_CONFIG

