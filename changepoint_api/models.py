"""HTTP and service-layer data contracts."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DetectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: list[float] = Field(
        ..., min_length=1, description="One-dimensional offline observations."
    )
    penalty: float = Field(
        ..., ge=0.0, description="Non-negative cost added for every segment."
    )
    min_segment_length: int = Field(
        default=1, ge=1, description="Minimum number of samples in a segment."
    )
    algorithm: Literal["exact_dp", "pruned_dp"] = "exact_dp"

    @field_validator("samples")
    @classmethod
    def validate_finite_samples(cls, value: list[float]) -> list[float]:
        for index, sample in enumerate(value):
            if not math.isfinite(sample):
                raise ValueError(
                    f"samples[{index}] must be finite, got {sample}"
                )
        return value

    @field_validator("penalty")
    @classmethod
    def validate_finite_penalty(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("penalty must be finite")
        return value


class SegmentResult(BaseModel):
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=1)
    length: int = Field(..., ge=1)
    mean: float
    cost: float = Field(..., description="Within-segment sum of squared errors.")
    penalty: float = Field(..., ge=0.0)


class NumericGuard(BaseModel):
    prefix_strategy: str
    final_check: str
    prefix_anchor: float
    max_abs_error: float
    max_relative_error: float
    clamped_negative_costs: int
    candidate_evaluations: int


class DetectionResponse(BaseModel):
    run_id: str
    status: Literal["success"] = "success"
    input_digest: str
    algorithm: Literal["exact_dp", "pruned_dp"]
    package_version: str
    algorithm_version: str
    kernel_version: str
    tie_break_rule: str
    n: int
    penalty: float
    min_segment_length: int
    change_points: list[int]
    num_segments: int
    segments: list[SegmentResult]
    total_sse: float
    total_penalty: float
    objective: float
    numeric_guard: NumericGuard
    duration_seconds: float = Field(..., ge=0.0)
    created_at: str
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ErrorBody(BaseModel):
    error_code: str
    message: str
    run_id: str | None = None


class RunError(BaseModel):
    category: str
    message: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: Literal["running", "success", "failed"]
    input_digest: str
    algorithm: Literal["exact_dp", "pruned_dp"]
    result: DetectionResponse | None = None
    error: RunError | None = None
    created_at: str


class EventResponse(BaseModel):
    seq: int
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    event_type: str
    message: str
    details: dict[str, Any]
    created_at: str


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
