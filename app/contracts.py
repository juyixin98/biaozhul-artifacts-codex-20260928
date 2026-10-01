"""Stable request/response contracts for the survival analysis API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError


CiMethod = Literal["greenwood_linear", "log_log"]


class ObservationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(..., min_length=1, max_length=100)
    group: str = Field(..., min_length=1, max_length=100)
    time: float
    event: bool
    entry: float = Field(default=0.0)

    @field_validator("time", "entry")
    @classmethod
    def validate_finite_non_negative(cls, value: float, info) -> float:
        if not value == value:
            raise PydanticCustomError("NON_FINITE_VALUE", f"{info.field_name} must be finite")
        if value in (float("inf"), float("-inf")):
            raise PydanticCustomError("NON_FINITE_VALUE", f"{info.field_name} must be finite")
        if value < 0:
            raise PydanticCustomError(
                "NEGATIVE_TIME", f"{info.field_name} must be greater than or equal to zero"
            )
        return float(value)

    @model_validator(mode="after")
    def validate_time_window(self) -> "ObservationIn":
        if self.time <= 0:
            raise PydanticCustomError(
                "NON_POSITIVE_TIME", "time must be strictly greater than zero"
            )
        if self.entry >= self.time:
            raise PydanticCustomError(
                "ENTRY_NOT_BEFORE_TIME", "entry must be strictly before time"
            )
        return self


class SurvivalAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_request_id: str | None = Field(default=None, min_length=1, max_length=100)
    analysis_name: str | None = Field(default=None, min_length=1, max_length=200)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    significance_level: float = Field(default=0.05, gt=0.0, lt=1.0)
    confidence_interval_method: CiMethod = "greenwood_linear"
    groups: list[str] | None = Field(default=None, min_length=1)
    observations: list[ObservationIn] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_groups_and_observations(self) -> "SurvivalAnalysisRequest":
        observation_ids: set[str] = set()
        observed_groups: set[str] = set()
        for observation in self.observations:
            if observation.observation_id in observation_ids:
                raise PydanticCustomError(
                    "DUPLICATE_OBSERVATION_ID",
                    f"duplicate observation_id: {observation.observation_id}",
                )
            observation_ids.add(observation.observation_id)
            observed_groups.add(observation.group)

        if self.groups is not None:
            declared: set[str] = set()
            for group in self.groups:
                if not group.strip():
                    raise PydanticCustomError("BLANK_GROUP", "group labels must not be blank")
                if group in declared:
                    raise PydanticCustomError(
                        "DUPLICATE_GROUP", f"duplicate group in groups: {group}"
                    )
                declared.add(group)
            unknown = sorted(observed_groups - declared)
            if unknown:
                raise PydanticCustomError(
                    "UNKNOWN_GROUP",
                    "observations contain groups not declared in groups",
                    {"unknown_groups": unknown},
                )
        return self


class SyntheticDatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=311311, ge=0, le=2**32 - 1)
    sample_size_per_group: int = Field(default=24, ge=1, le=1000)
    delayed_entry: bool = False
    censor_low: float = Field(default=4.0)
    censor_high: float = Field(default=18.0)

    @model_validator(mode="after")
    def validate_censor_window(self) -> "SyntheticDatasetRequest":
        if self.censor_low <= 0 or self.censor_high <= self.censor_low:
            raise PydanticCustomError(
                "INVALID_CENSOR_WINDOW",
                "censor_low must be positive and censor_high must be greater",
            )
        return self


class Failure(BaseModel):
    code: str
    message: str
    affects: str


class Notice(BaseModel):
    code: str
    message: str
    affects: str


class TraceEntry(BaseModel):
    timestamp: str
    component: str
    step: str
    status: Literal["ok", "warning", "skipped", "estimated", "not_run", "failed"]
    detail: dict


class SurvivalPoint(BaseModel):
    time: float
    kind: Literal["baseline", "event", "censoring", "entry"]
    risk_before: int
    entered: int
    events: int
    censored: int
    risk_after: int
    survival: float | None
    greenwood_variance: float | None
    ci_low: float | None
    ci_high: float | None
    ci_clipped: bool


class GroupResult(BaseModel):
    group: str
    n: int
    events: int
    censored: int
    median_survival_time: float | None
    median_status: Literal["estimated", "not_reached", "not_estimable"]
    points: list[SurvivalPoint]


class LogRankComparison(BaseModel):
    status: Literal["estimated", "not_run", "failed"]
    reason: str | None = None
    method: str = "log_rank"
    group_order: list[str]
    observed: dict[str, int]
    expected: dict[str, float | None]
    variance_covariance_matrix: list[list[float | None]] | None = None
    chi_square: float | None = None
    degrees_freedom: int | None = None
    p_value: float | None = None


class SurvivalAnalysisResponse(BaseModel):
    schema_version: str
    service_version: str
    kernel_version: str
    analysis_id: str
    request_id: str
    analysis_name: str | None
    created_at: str
    processing_location: str
    config: dict
    summary: dict
    groups: list[GroupResult]
    comparison: LogRankComparison
    trace: list[TraceEntry]
    notices: list[Notice]
    failures: list[Failure]
    disclaimer: str


class ErrorResponse(BaseModel):
    schema_version: str
    service_version: str
    request_id: str
    error: dict
    failures: list[Failure]


class EventLogRecord(BaseModel):
    sequence: int
    timestamp: str
    component: str
    step: str
    status: str
    detail: dict


class AnalysisEventsResponse(BaseModel):
    analysis_id: str
    request_id: str
    events: list[EventLogRecord]
