"""Pydantic contracts for samples, requests, and versioned results."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Sample(BaseModel):
    """One weighted binary outcome and predicted probability."""

    id: str = Field(min_length=1)
    y: int = Field(ge=0, le=1)
    p: float
    w: float = Field(default=1.0)

    @field_validator("p")
    @classmethod
    def validate_probability(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("p must be finite and within [0, 1]; clipping is not allowed")
        return value

    @field_validator("w")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("w must be finite and strictly positive")
        return value


class FixedBinningConfig(BaseModel):
    n_bins: int = Field(default=10, ge=1, le=100)
    edges: list[float] | None = Field(default=None)

    @model_validator(mode="after")
    def validate_edges(self) -> "FixedBinningConfig":
        if self.edges is None:
            return self
        if len(self.edges) < 2:
            raise ValueError("fixed bin edges must contain at least two values")
        if not all(math.isfinite(x) for x in self.edges):
            raise ValueError("fixed bin edges must be finite")
        if self.edges[0] != 0.0 or self.edges[-1] != 1.0:
            raise ValueError("fixed bin edges must start at 0 and end at 1")
        if any(left >= right for left, right in zip(self.edges, self.edges[1:])):
            raise ValueError("fixed bin edges must be strictly increasing")
        self.n_bins = len(self.edges) - 1
        return self


class QuantileBinningConfig(BaseModel):
    n_bins: int = Field(default=10, ge=1, le=100)


class CalibratorConfig(BaseModel):
    type: Literal["identity", "isotonic"] = "isotonic"


class RunRequest(BaseModel):
    run_id: str | None = Field(default=None, min_length=1)
    fit: list[Sample]
    evaluation: list[Sample] = Field(alias="eval")
    calibrator: CalibratorConfig = Field(default_factory=CalibratorConfig)
    fixed_bins: FixedBinningConfig = Field(default_factory=FixedBinningConfig)
    quantile_bins: QuantileBinningConfig = Field(default_factory=QuantileBinningConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def validate_sample_sets(self) -> "RunRequest":
        if not self.fit:
            raise ValueError("fit must contain at least one sample")
        if not self.evaluation:
            raise ValueError("eval must contain at least one sample")

        fit_ids = [sample.id for sample in self.fit]
        eval_ids = [sample.id for sample in self.evaluation]
        if len(set(fit_ids)) != len(fit_ids):
            raise ValueError("fit sample ids must be unique")
        if len(set(eval_ids)) != len(eval_ids):
            raise ValueError("eval sample ids must be unique")
        overlap = sorted(set(fit_ids) & set(eval_ids))
        if overlap:
            raise ValueError(f"fit and eval sample ids must be disjoint; overlap={overlap}")
        return self


class CheckResult(BaseModel):
    name: str
    passed: bool
    observed: float
    expected: float
    absolute_tolerance: float
    details: dict[str, Any] = Field(default_factory=dict)


class BinRecord(BaseModel):
    index: int
    scheme: Literal["fixed", "equal_frequency"]
    interval: str
    range_lower: float | None
    range_upper: float | None
    is_empty: bool
    count: int
    total_weight: float
    weight_fraction: float
    effective_sample_size: float
    p_min: float | None
    p_max: float | None
    p_mean: float | None
    y_mean: float | None


class BinningStatistics(BaseModel):
    scheme: Literal["fixed", "equal_frequency"]
    n_bins_requested: int
    n_nonempty_bins: int
    n_empty_bins: int
    tie_policy: str
    sample_count: int
    total_weight: float
    effective_sample_size: float
    outcome_rate: float
    mean_probability: float
    brier: float
    mean_forecast_variance_within_bin: float
    reliability: float
    resolution: float
    uncertainty: float
    relaxed_brier_from_components: float
    decomposition_residual: float
    bins: list[BinRecord]
    weight_conservation_check: CheckResult
    count_conservation_check: CheckResult
    decomposition_check: CheckResult


class SampleSetSummary(BaseModel):
    count: int
    total_weight: float
    effective_sample_size: float
    sample_ids: list[str]


class CalibratorInfo(BaseModel):
    type: Literal["identity", "isotonic"]
    fit_summary: SampleSetSummary
    evaluation_summary: SampleSetSummary
    disjoint_fit_and_evaluation: bool
    overlap_ids: list[str]
    parameters: dict[str, Any]


class VersionInfo(BaseModel):
    schema_version: str
    kernel_version: str
    generated_at: str
    python: str
    numpy: str
    scipy: str
    fastapi: str
    pydantic: str


class RunResult(BaseModel):
    run_id: str
    status: Literal["SUCCEEDED"]
    input_hash: str
    versions: VersionInfo
    calibrator: CalibratorInfo
    analyses: dict[
        Literal["raw", "calibrated"],
        dict[Literal["fixed", "equal_frequency"], BinningStatistics],
    ]
    checks: list[CheckResult]
    metadata: dict[str, Any]


class RunSummary(BaseModel):
    run_id: str
    status: str
    schema_version: str
    input_hash: str
    created_at: str


class ErrorBody(BaseModel):
    status: Literal["FAILED"] = "FAILED"
    category: str
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
