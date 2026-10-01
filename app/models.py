from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Method = Literal["ledoit_wolf", "robust_tyler_shrinkage"]


class CovarianceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data: list[list[float | None]] = Field(
        ..., description="Observations in rows and variables in columns."
    )
    method: Method = "ledoit_wolf"
    column_scales: list[float] | None = Field(
        default=None,
        alias="columnScales",
        description="Optional fixed positive standard scales. MAD is used when omitted.",
    )
    assume_centered: bool = Field(
        default=False,
        alias="assumeCentered",
        description="Use zero as location. Tyler's estimator assumes this.",
    )
    shrinkage: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Fixed Tyler shrinkage in [0, 1]."
    )
    max_iterations: int = Field(
        default=200, alias="maxIterations", ge=1, le=2000
    )
    tolerance: float = Field(default=1e-9, alias="tolerance", gt=0.0, lt=0.1)

    @field_validator("data")
    @classmethod
    def _validate_rows(cls, value: list[list[float | None]]) -> list[list[float | None]]:
        if not value:
            raise ValueError("at least one observation is required")
        width = len(value[0])
        if width == 0:
            raise ValueError("at least one variable is required")
        for i, row in enumerate(value):
            if len(row) != width:
                raise ValueError(
                    f"row {i} has {len(row)} values; expected {width}"
                )
        return value

    @model_validator(mode="after")
    def _validate_scales(self) -> "CovarianceRequest":
        if self.column_scales is not None:
            if len(self.column_scales) != len(self.data[0]):
                raise ValueError("columnScales length must equal data width")
            if any(value <= 0.0 for value in self.column_scales):
                raise ValueError("columnScales values must be strictly positive")
        return self


class ErrorDetail(BaseModel):
    code: str
    message: str
    location: str | None = None


class CovarianceResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field(alias="schemaVersion")
    estimator_version: str = Field(alias="estimatorVersion")
    request_id: str = Field(alias="requestId")
    status: Literal["ok", "uncertain", "error"]
    method: Method
    n_samples: int = Field(alias="nSamples")
    n_dimensions: int = Field(alias="nDimensions")
    location: list[float]
    covariance: list[list[float]] | None = None
    eigenvalues: list[float]
    condition_number: float | None = Field(default=None, alias="conditionNumber")
    rank: int
    invertible: bool
    shrinkage: float
    outlier_weights: list[float] = Field(alias="outlierWeights")
    missing_mask: list[list[bool]] = Field(alias="missingMask")
    imputed_values: list[list[float | None]] = Field(alias="imputedValues")
    column_scales: list[float] = Field(alias="columnScales")
    assumptions: list[str]
    steps: list[dict[str, Any]]
    failures: list[ErrorDetail]
    uncertainties: list[ErrorDetail]


class HealthResponse(BaseModel):
    status: str
    request_id: str = Field(alias="requestId")
    database: str
