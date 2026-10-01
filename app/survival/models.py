"""Typed contracts shared by the statistical core and API layer."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


ConfidenceIntervalMethod = Literal["log-log", "linear"]


class Severity(str, Enum):
    WARNING = "warning"
    UNCERTAIN = "uncertain"


class ObservationIn(BaseModel):
    """One right-censored observation, optionally left-truncated."""

    time: float = Field(..., ge=0, description="Observation/event time on an absolute time scale.")
    event: bool = Field(..., description="True for an observed event; false for right censoring.")
    entry_time: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Absolute time at which the subject enters the risk set (left "
            "truncation/delayed entry). At a tied event time, subjects already "
            "in the interval are processed and the new subject enters afterward."
        ),
    )
    group: str = Field(..., min_length=1, description="Non-empty group label.")

    @field_validator("group")
    @classmethod
    def _non_empty_group(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("group must not be empty or whitespace")
        return value

    @model_validator(mode="after")
    def _validate_times(self) -> "ObservationIn":
        if self.entry_time >= self.time:
            raise ValueError("entry_time must be strictly smaller than time")
        if self.time == 0:
            raise ValueError("time zero is reserved for the pre-study baseline row")
        return self


class KMRequest(BaseModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    observations: list[ObservationIn] = Field(..., min_length=1)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    confidence_interval_method: ConfidenceIntervalMethod = "log-log"
    include_censor_times: bool = True
    include_entry_times: bool = True


class LogRankRequest(BaseModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    observations: list[ObservationIn] = Field(..., min_length=1)


class RiskPoint(BaseModel):
    time: float
    entered: int
    at_risk: int
    events: int
    censored: int
    survival: float | None = None
    variance: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    event_point: bool


class MedianSummary(BaseModel):
    reachable: bool
    value: float | None = None
    interval_lower: float | None = None
    interval_upper: float | None = None
    status: Literal["estimated", "not_reached", "empty_group", "no_events", "not_identifiable"]
    reason: str


class GroupKMResult(BaseModel):
    group: str
    n: int
    events: int
    censored: int
    left_truncated: bool
    timeline: list[RiskPoint]
    median: MedianSummary


class Notice(BaseModel):
    code: str
    severity: Severity
    message: str
    groups: list[str] = Field(default_factory=list)


class KMResult(BaseModel):
    groups: list[GroupKMResult]
    notices: list[Notice] = Field(default_factory=list)


class EventContribution(BaseModel):
    time: float
    group: str
    observed: int
    expected: float
    variance_diagonal: float


class LogRankGroupStatistic(BaseModel):
    group: str
    observed: int
    expected: float
    observed_minus_expected: float


class LogRankResult(BaseModel):
    groups: list[str]
    degrees_of_freedom: int
    chi_square: float
    p_value: float | None
    testable: bool
    status: Literal["valid", "not_testable"]
    event_contributions: list[EventContribution]
    group_statistics: list[LogRankGroupStatistic]
    notices: list[Notice] = Field(default_factory=list)
