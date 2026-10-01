"""HTTP request and response contracts."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateRunRequest(StrictModel):
    run_id: str = Field(min_length=1, max_length=200)
    alpha: float = Field(gt=0.0)
    beta: float = Field(gt=0.0)

    @field_validator("alpha", "beta", mode="before")
    @classmethod
    def validate_prior(cls, value: object) -> object:
        if isinstance(value, (bool, str)):
            raise ValueError("prior parameters must be numeric")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("prior parameters must be finite")
        return value


class BatchRequest(StrictModel):
    successes: int = Field(ge=0)
    failures: int = Field(ge=0)
    reason: str | None = Field(default=None, max_length=500)

    @field_validator("successes", "failures", mode="before")
    @classmethod
    def validate_counts(cls, value: object) -> object:
        if isinstance(value, (bool, str)):
            raise ValueError("counts must be non-string integers")
        return value


class BatchResponse(StrictModel):
    run_id: str
    batch_id: str
    version: int
    batch_version_id: int
    run_version_id: int
    sequence: int
    event_type: Literal["batch_submitted", "batch_revised"]
    successes: int
    failures: int
    reason: str | None
    posterior_alpha: float
    posterior_beta: float
    total_successes: int
    total_failures: int
    created_at: str


class RunResponse(StrictModel):
    run_id: str
    alpha: float
    beta: float
    created_at: str


class BatchSnapshot(StrictModel):
    batch_id: str
    version: int
    successes: int
    failures: int
    status: Literal["active", "superseded"]
    reason: str | None
    created_at: str


class RunVersionResponse(StrictModel):
    run_id: str
    run_version_id: int
    sequence: int
    event_type: Literal["created", "batch_submitted", "batch_revised"]
    batch_version_id: int | None
    posterior_alpha: float
    posterior_beta: float
    total_successes: int
    total_failures: int
    created_at: str
    metadata: dict[str, Any]


class IntervalResponse(StrictModel):
    lower: float
    upper: float
    level: float


class PosteriorSummaryResponse(StrictModel):
    run_id: str
    prior_alpha: float
    prior_beta: float
    posterior_alpha: float
    posterior_beta: float
    total_successes: int
    total_failures: int
    mean: float
    variance: float
    credible_interval: IntervalResponse
    confidence_interval: IntervalResponse | None = None
    active_batches: list[BatchSnapshot]


class PredictionPointResponse(StrictModel):
    future_successes: int
    probability: float


class PredictionRequest(StrictModel):
    future_n: int = Field(ge=0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)

    @field_validator("future_n", mode="before")
    @classmethod
    def validate_future_n(cls, value: object) -> object:
        if isinstance(value, (bool, str)):
            raise ValueError("future_n must be a non-string integer")
        return value

    @field_validator("confidence_level", mode="before")
    @classmethod
    def validate_level(cls, value: object) -> object:
        if isinstance(value, (bool, str)):
            raise ValueError("confidence_level must be numeric")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("confidence_level must be finite")
        return value


class PredictionResponse(StrictModel):
    run_id: str
    future_n: int
    posterior_alpha: float
    posterior_beta: float
    points: list[PredictionPointResponse]
    probability_sum: float
    mean: float
    variance: float
    credible_interval: IntervalResponse
    confidence_interval: IntervalResponse | None = None


class ErrorResponse(BaseModel):
    error: str
    category: Literal[
        "invalid_input",
        "state_conflict",
        "resource_exhausted",
        "computation_failure",
    ]
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None
