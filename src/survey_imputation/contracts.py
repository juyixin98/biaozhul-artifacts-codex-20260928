"""Pydantic contracts and plain dataclasses used by the statistical kernel.

Records intentionally use ``rating: float | None``. A numeric zero is a real
structural zero and is never converted to missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import InvalidParameterError
from .versioning import CONTRACT_VERSION, CORE_VERSION


class MissingnessMechanism(StrEnum):
    complete = "complete"
    mcar = "mcar"
    mar = "mar"
    mnar = "mnar"


class DiagnosticStatus(StrEnum):
    accepted = "accepted"
    rejected = "rejected"
    undetermined = "undetermined"


class SurveyRecord(BaseModel):
    record_id: str = Field(min_length=1)
    age_z: float
    member: Literal[0, 1]
    used_support: Literal[0, 1]
    contact_frequency: float = Field(ge=0.0, le=10.0)
    rating: float | None = Field(default=None, ge=0.0, le=10.0)

    @field_validator("record_id")
    @classmethod
    def record_id_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("record_id must not be blank")
        return value


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    request_id: str | None = Field(default=None, min_length=1)
    records: list[SurveyRecord] = Field(min_length=1)
    method: Literal["complete_data", "complete_case", "mean_imputation", "multiple_imputation"]
    m: int = Field(default=20, ge=2, le=200)
    seed: int = Field(default=20260930, ge=0, le=2**63 - 1)


class GenerateDatasetRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    request_id: str | None = Field(default=None, min_length=1)
    n: int = Field(default=1000, ge=20, le=10_000)
    mechanism: MissingnessMechanism = MissingnessMechanism.complete
    missing_probability: float = Field(default=0.20, gt=0.0, lt=1.0)
    seed: int = Field(default=20260930, ge=0, le=2**63 - 1)

    @field_validator("missing_probability")
    @classmethod
    def valid_probability_for_scale(cls, value: float) -> float:
        if not 0.05 <= value <= 0.60:
            raise ValueError("missing_probability must be between 0.05 and 0.60")
        return value


class PointResult(BaseModel):
    method: str
    model_version: str
    n_records: int
    n_missing: int
    estimate: float
    variance: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    degrees_of_freedom: float | None = None
    imputation_count: int | None = None
    marked_imputed: int = 0
    note: str


class ImputedValue(BaseModel):
    record_id: str
    imputation_index: int
    rating: float


class AnalysisResponse(BaseModel):
    contract_version: str = CONTRACT_VERSION
    core_version: str = CORE_VERSION
    request_id: str
    estimate: PointResult
    diagnostics: list["Diagnostic"]
    imputations_preview: list[ImputedValue] = []


class Diagnostic(BaseModel):
    code: str
    status: DiagnosticStatus
    message: str
    observed_state: dict[str, int | float | str | bool | None]
    record_id: str | None = None


class GeneratedDatasetResponse(BaseModel):
    contract_version: str = CONTRACT_VERSION
    request_id: str
    mechanism: MissingnessMechanism
    seed: int
    n_records: int
    n_missing: int
    true_mean_rating: float
    complete_baseline_mean: float
    records: list[SurveyRecord]
    diagnostics: list[Diagnostic]


@dataclass(frozen=True)
class DesignMatrices:
    """Validated arrays used inside the kernel.

    ``y`` has NaN for missing observations; ``mask`` is one for missing. The
    pair is cross-checked so structural zeros remain observed.
    """

    record_ids: list[str]
    y: "np.ndarray"
    x: "np.ndarray"
    mask: "np.ndarray"
    feature_names: tuple[str, ...]


def validate_imputation_options(method: str, m: int, seed: int) -> None:
    if method == "multiple_imputation" and m < 2:
        raise InvalidParameterError("multiple imputation requires at least two imputations")
    if seed < 0:
        raise InvalidParameterError("seed must be non-negative")


AnalysisResponse.model_rebuild()
