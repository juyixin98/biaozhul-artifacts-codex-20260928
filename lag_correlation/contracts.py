"""Pydantic request/response contracts for the HTTP service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .config import MAX_LAG_POINTS


class SeriesPayload(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    times: list[float] = Field(min_length=2)
    values: list[float]

    @model_validator(mode="after")
    def _validate_shape_and_time(self) -> "SeriesPayload":
        if len(self.times) != len(self.values):
            raise ValueError("times and values must have equal length")
        if any(not _is_finite_number(t) for t in self.times):
            raise ValueError("times must contain finite numbers")
        if any(self.times[index] <= self.times[index - 1] for index in range(1, len(self.times))):
            raise ValueError("times must be strictly increasing")
        if any(not _is_finite_number(v) for v in self.values):
            raise ValueError("values must contain finite numbers")
        return self


class LagGrid(BaseModel):
    min_lag: float = Field(ge=-1.0e12)
    max_lag: float = Field(le=1.0e12)
    step: float = Field(gt=0.0, le=1.0e12)

    @model_validator(mode="after")
    def _validate_range(self) -> "LagGrid":
        if self.min_lag > self.max_lag:
            raise ValueError("min_lag must be less than or equal to max_lag")
        point_count = int(round((self.max_lag - self.min_lag) / self.step)) + 1
        if point_count > MAX_LAG_POINTS:
            raise ValueError(f"lag grid must contain at most {MAX_LAG_POINTS} points")
        return self


class MatchingWindow(BaseModel):
    tolerance: float = Field(gt=0.0, le=1.0e12)


class AnalysisRequest(BaseModel):
    request_id: str | None = Field(default=None, max_length=128)
    x: SeriesPayload
    y: SeriesPayload
    lag_grid: LagGrid
    matching: MatchingWindow


class PairDetail(BaseModel):
    anchor_side: Literal["x", "y"]
    anchor_time: float
    paired_time: float
    anchor_index: int
    paired_indices: list[int]
    time_delta: float
    lag: float


class LagResult(BaseModel):
    lag: float
    n_pairs: int
    correlation: float | None
    accepted: bool
    status: str
    reason: str
    pair_count_from_anchor: int
    dense_samples_used: int
    dense_samples_unmatched: int
    pair_details: list[PairDetail] | None = None


class AnalysisResponse(BaseModel):
    request_id: str
    status: Literal["ok", "unknown"]
    failure_category: str | None
    reason: str
    peak_lag: float | None
    peak_correlation: float | None
    anchor_side: Literal["x", "y"]
    valid_lag_count: int
    evaluated_lag_count: int
    minimum_pairs: int
    detrend: str
    standardization: str
    schema_version: str
    algorithm_version: str
    matching_rule: str
    causality_notice: str
    best_lag: LagResult | None
    diagnostics: list[dict]


class HealthResponse(BaseModel):
    service: str
    status: str
    schema_version: str
    algorithm_version: str


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value not in (
        float("inf"),
        float("-inf"),
    ) and value == value
