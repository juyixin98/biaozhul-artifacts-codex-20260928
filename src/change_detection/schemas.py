from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(min_length=1)
    min_segment_length: int = Field(default=1, ge=1)
    penalty: float = Field(default=0.0, ge=0.0)
    prune: bool = True
    verify_pruning: bool = True

    @field_validator("values")
    @classmethod
    def values_must_be_finite(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all values must be finite")
        return values

    @field_validator("penalty")
    @classmethod
    def penalty_must_be_finite(cls, penalty: float) -> float:
        if not math.isfinite(penalty):
            raise ValueError("penalty must be finite")
        return penalty


class SegmentResponse(BaseModel):
    start: int
    end: int
    mean: float
    cost: float


class AnalyzeResponse(BaseModel):
    run_id: str
    schema_version: int
    core_version: str
    input_sha256: str
    status: Literal["SUCCEEDED"]
    objective: float
    sse: float
    penalty_per_segment: float
    total_penalty: float
    segment_count: int
    breakpoints: list[int]
    tie_rule: str
    solution_mode: str
    verification_mode: str
    transition_count: int
    pruned_transitions: int
    segments: list[SegmentResponse]


class EventResponse(BaseModel):
    event: str
    payload: dict
    created_at: str


class RunResponse(BaseModel):
    run_id: str
    schema_version: int
    core_version: str
    status: Literal["RUNNING", "SUCCEEDED", "FAILED"]
    input_sha256: str
    request: AnalyzeRequest
    result: AnalyzeResponse | None
    error_category: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    events: list[EventResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    core_version: str
    schema_version: int


class ErrorResponse(BaseModel):
    category: str
    message: str
    details: dict
