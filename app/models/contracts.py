from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field


def _reject_bool(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("boolean values are not accepted as numeric counts or parameters")
    return value


def _finite_positive(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("value must be a finite positive number")
    return value


def _non_negative_integer(value: int) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError("value must be a finite non-negative integer")
    return value


def _optional_text(value: str | None) -> str | None:
    if value is not None and isinstance(value, bool):
        raise ValueError("reason must be text or null, not a boolean")
    return value


PriorParameter = Annotated[
    float,
    BeforeValidator(_reject_bool),
    Field(gt=0),
    AfterValidator(_finite_positive),
]
Count = Annotated[
    int,
    BeforeValidator(_reject_bool),
    Field(strict=True, ge=0),
    AfterValidator(_non_negative_integer),
]
OptionalReason = Annotated[
    str | None,
    BeforeValidator(_optional_text),
    Field(default=None, max_length=500),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CreateModelRequest(StrictModel):
    alpha: PriorParameter
    beta: PriorParameter
    reason: OptionalReason = None


class BatchCountsRequest(StrictModel):
    success_count: Count
    failure_count: Count
    reason: OptionalReason = None


class RevisionRequest(BatchCountsRequest):
    pass


class PredictionRequest(StrictModel):
    trials: Count = 0
    include_distribution: bool = False
    reason: OptionalReason = None


class CredibleInterval(BaseModel):
    level: float
    lower: float
    upper: float
    interpretation: Literal["Bayesian credible interval"]


class FrequentistConfidenceInterval(BaseModel):
    level: float
    lower: float | None
    upper: float | None
    method: Literal["Wilson score confidence interval (frequentist)"]
    interpretation: Literal["Frequentist confidence interval, not a Bayesian credible interval"]


class PosteriorSummary(BaseModel):
    model_id: str
    version: int
    prior_alpha: float
    prior_beta: float
    success_count: int
    failure_count: int
    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    posterior_variance: float
    credible_interval: CredibleInterval
    frequentist_confidence_interval: FrequentistConfidenceInterval


class ModelResponse(StrictModel):
    id: str
    version: int
    prior_alpha: float
    prior_beta: float
    success_count: int
    failure_count: int
    created_at: str


class BatchResponse(StrictModel):
    id: str
    model_id: str
    revision_of_batch_id: str | None
    status: Literal["active", "superseded", "reverted"]
    version: int
    success_count: int
    failure_count: int
    reason: str | None
    created_at: str


class BatchRevisionResponse(BatchResponse):
    withdraw_version: int
    add_version: int


class EventResponse(StrictModel):
    model_id: str
    version: int
    event_type: str
    batch_id: str | None
    delta_success: int
    delta_failure: int
    total_success: int
    total_failure: int
    details: dict
    created_at: str


class PredictionResponse(StrictModel):
    posterior: PosteriorSummary
    trials: int
    probabilities: list[float] | None
    probability_sum: float
    distribution_tolerance: float
