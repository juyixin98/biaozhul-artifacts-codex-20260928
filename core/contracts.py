"""Pydantic contracts shared by the statistical core, API, and persistence layer."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .errors import ProbabilityRangeError, SampleContractError, WeightError


BinningMethod = Literal["fixed", "quantile"]
RunStatus = Literal["success", "failed"]


class SampleBatch(BaseModel):
    """One binary observation per forecast probability.

    Weights are nonnegative survey/sample weights. Zero-weight observations remain
    part of the input length and count but have no statistical weight.
    """

    y: list[int] = Field(..., min_length=1)
    p: list[float] = Field(..., min_length=1)
    w: Optional[list[float]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_raw_contract(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        y = data.get("y", [])
        p = data.get("p", [])
        w = data.get("w")

        bad_y = sorted({item for item in y if item not in (0, 1)})
        if bad_y:
            raise SampleContractError(f"outcomes must be binary; found {bad_y[:3]}")
        bad_p = [item for item in p if not (0.0 <= float(item) <= 1.0)]
        if bad_p:
            raise ProbabilityRangeError(
                "probabilities must be in the closed interval [0, 1]; "
                f"first invalid value={bad_p[0]!r}"
            )
        if w is not None:
            bad_w = [item for item in w if float(item) < 0.0]
            if bad_w:
                raise WeightError(f"weights must be nonnegative; first invalid weight={bad_w[0]!r}")
        if len(y) != len(p):
            raise SampleContractError("y and p must contain the same number of samples")
        if w is not None and len(w) != len(y):
            raise SampleContractError("w must contain the same number of samples as y and p")
        total_weight = sum(w if w is not None else [1.0] * len(y))
        if total_weight <= 0.0:
            raise WeightError("total sample weight must be positive")
        return data


class AnalysisRequest(SampleBatch):
    method: BinningMethod = "fixed"
    n_bins: int = Field(default=10, ge=1, le=1000)
    edges: Optional[list[float]] = None
    run_label: Optional[str] = Field(default=None, max_length=200)

    @model_validator(mode="before")
    @classmethod
    def validate_raw_analysis(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if data.get("method", "fixed") == "fixed" and data.get("edges") is None and int(data.get("n_bins", 10)) < 1:
            raise SampleContractError("fixed binning requires at least one bin")
        edges = data.get("edges")
        if edges is not None:
            if len(edges) < 2:
                raise SampleContractError("edges must contain at least lower and upper boundaries")
            if edges[0] != 0.0 or edges[-1] != 1.0:
                raise SampleContractError("edges must start at 0.0 and end at 1.0")
            if any(left >= right for left, right in zip(edges, edges[1:])):
                raise SampleContractError("edges must be strictly increasing")
        return data


class CalibrationFitRequest(BaseModel):
    train: SampleBatch
    evaluate: Optional[SampleBatch] = None
    method: BinningMethod = "quantile"
    n_bins: int = Field(default=10, ge=1, le=1000)
    train_edges: Optional[list[float]] = None
    evaluation_edges: Optional[list[float]] = None
    run_label: Optional[str] = Field(default=None, max_length=200)


class RawSampleBatch(BaseModel):
    y: list
    p: list
    w: Optional[list] = None


class RawAnalysisRequest(BaseModel):
    y: list
    p: list
    w: Optional[list] = None
    method: BinningMethod = "fixed"
    n_bins: int = 10
    edges: Optional[list] = None
    run_label: Optional[str] = None


class RawCalibrationFitRequest(BaseModel):
    train: RawSampleBatch
    evaluate: Optional[RawSampleBatch] = None
    method: BinningMethod = "quantile"
    n_bins: int = 10
    train_edges: Optional[list] = None
    evaluation_edges: Optional[list] = None
    run_label: Optional[str] = None


class BinStats(BaseModel):
    bin_index: int
    lower: float
    upper: float
    count: int
    zero_weight_count: int
    weight: float
    weight_fraction: float
    effective_sample_size: float
    observed_rate: float
    mean_prediction: float
    brier: float
    binned_brier: float
    empty: bool


class Decomposition(BaseModel):
    raw_brier: float
    binned_brier: float
    reliability: float
    resolution: float
    uncertainty: float
    decomposition_residual: float
    identity_residual: float


class WeightSummary(BaseModel):
    count: int
    total_weight: float
    effective_sample_size: float
    sum_weight_fractions: float


class AnalysisResult(BaseModel):
    run_id: str
    status: Literal["success"] = "success"
    method: BinningMethod
    requested_bins: int
    occupied_bins: int
    empty_bins: int
    edges: list[float]
    bins: list[BinStats]
    decomposition: Decomposition
    weights: WeightSummary
    base_rate: float
    mean_prediction: float
    calculation_trace: list[str]


class CalibrationSummary(BaseModel):
    edges: list[float]
    bin_predicted_rates: list[Optional[float]]
    occupied_bins: int
    empty_bins: int


class CalibrationResult(BaseModel):
    run_id: str
    status: Literal["success"] = "success"
    calibration: CalibrationSummary
    train_analysis: AnalysisResult
    evaluation_analysis: Optional[AnalysisResult] = None
    transformed_probabilities: Optional[list[float]] = None
    calculation_trace: list[str]
