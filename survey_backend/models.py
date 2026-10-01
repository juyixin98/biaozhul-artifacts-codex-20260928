from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MethodName = Literal["complete_case", "mean_imputation", "multiple_imputation"]
DiagnosticState = Literal["accepted", "rejected", "undetermined"]


class SurveyRecord(BaseModel):
    """One local synthetic participant.

    ``None`` is missing, while numeric zero is a genuine measured zero.
    ``eligible=False`` means the observation is not permitted to train or
    estimate model parameters, even if all values are present.
    """

    record_id: str = Field(min_length=1, max_length=128)
    q1_screen_hours: float | None = None
    q2_sleep_hours: float | None = None
    satisfaction: float | None = None
    eligible: bool = True

    model_config = ConfigDict(extra="forbid")

    @field_validator("q1_screen_hours", "q2_sleep_hours", "satisfaction")
    @classmethod
    def reject_non_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("values must be finite numbers or null")
        return value


class AnalysisRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=128)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    records: list[SurveyRecord] = Field(min_length=1)
    methods: list[MethodName] | None = None
    imputations: int | None = Field(default=None, ge=2, le=50)
    burn_in: int | None = Field(default=None, ge=1, le=200)
    confidence: float | None = Field(default=None, gt=0.0, lt=1.0)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def reject_duplicate_method_names(self) -> AnalysisRequest:
        if self.methods is not None and len(self.methods) != len(set(self.methods)):
            raise ValueError("methods must not contain duplicates")
        return self


class CoefficientResult(BaseModel):
    name: str
    estimate: float
    variance: float
    standard_error: float
    p_value: float
    ci_lower: float
    ci_upper: float


class PoolingDetail(BaseModel):
    parameter: str
    within_variance: float
    between_variance: float
    total_variance: float
    relative_increase_variance: float
    fraction_missing_information: float
    degrees_of_freedom: float | None


class ImputationRun(BaseModel):
    imputation_index: int
    stream_tag: str
    imputed_q1: int
    imputed_q2: int
    coefficients: list[CoefficientResult]
    residual_variance: float


class MethodResult(BaseModel):
    method: MethodName
    status: DiagnosticState
    used_observations: int
    dropped_ineligible: int
    missing_counts: dict[str, int]
    imputed_counts: dict[str, int]
    coefficients: list[CoefficientResult] = Field(default_factory=list)
    residual_variance: float | None = None
    pooling: list[PoolingDetail] = Field(default_factory=list)
    runs: list[ImputationRun] = Field(default_factory=list)
    diagnostics: list["Diagnostic"] = Field(default_factory=list)


class Diagnostic(BaseModel):
    record_id: str | None = None
    redacted_record_id: str | None = None
    request_id: str | None = None
    redacted_request_id: str | None = None
    state: DiagnosticState
    code: str
    message: str
    observed_state: dict[str, int | float | bool | str | None]


class AnalysisResponse(BaseModel):
    analysis_id: str
    version: int
    dataset_id: str
    request_id: str
    request_hash: str
    confidence: float
    seed: int
    imputations: int
    burn_in: int
    assumptions: list[str]
    results: list[MethodResult]


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str | None = None
    redacted_request_id: str | None = None
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class AnalysisSummary(BaseModel):
    analysis_id: str
    version: int
    dataset_id: str
    request_id: str
    created_at: str


class AnalysisGroupResponse(BaseModel):
    analysis_id: str
    latest_version: int
    versions: list[AnalysisSummary]


MethodResult.model_rebuild()
