"""FastAPI request and response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SeriesPayload(BaseModel):
    times: list[float] = Field(..., min_length=1)
    values: list[float] = Field(..., min_length=1)


class AnalysisRequestPayload(BaseModel):
    a: SeriesPayload
    b: SeriesPayload
    lags: list[float] = Field(..., min_length=1)
    tolerance: float = Field(..., gt=0)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    include_pairs: bool = True


class PairPayload(BaseModel):
    a_index: int
    b_index: int
    a_time: float
    b_time: float
    a_value: float
    b_value: float
    aligned_b_time: float
    residual: float


class DiagnosisPayload(BaseModel):
    lag: float
    status: Literal["accepted", "rejected", "unknown"]
    reason_code: str | None
    reason: str | None
    a_count: int
    b_count: int
    candidate_edges: int
    selected_pairs: int
    pair_fraction: float
    selected_a_fraction: float
    selected_b_fraction: float
    selected_time_span: float
    aligned_overlap_span: float
    largest_gap: float
    mean_abs_residual: float | None
    max_abs_residual: float | None
    a_constant: bool
    b_constant: bool
    accepted: bool


class LagResultPayload(BaseModel):
    lag: float
    status: Literal["accepted", "rejected", "unknown"]
    correlation: float | None
    pair_count: int
    pairs: list[PairPayload]
    diagnosis: DiagnosisPayload
    reason_code: str | None
    reason: str | None


class ConfigPayload(BaseModel):
    detrend: str
    standardization: str
    matching: str
    min_pairs: int
    min_pair_fraction: float
    result_version: str


class AnalysisResponsePayload(BaseModel):
    request_id: str
    result_version: str
    config: ConfigPayload
    tolerance: float
    lags: list[float]
    results: list[LagResultPayload]
    peak_lag: float | None
    peak_correlation: float | None
    peak_kind: Literal["positive", "negative"] | None
    warning: str
    status_summary: dict[str, int]


class ErrorPayload(BaseModel):
    error_code: str
    message: str
    request_id: str | None = None
    details: dict | None = None
