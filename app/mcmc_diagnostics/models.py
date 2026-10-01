"""Pydantic sample contract and result schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DiagnosticLevel = Literal["info", "warning", "error", "inconclusive"]
DiagnosticStatus = Literal["ok", "warning", "inconclusive", "error"]


class SampleContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = Field(default=None, max_length=128)
    max_lag: int | None = Field(default=None, ge=0)
    include_acf: bool = True


class SingleChainRequest(SampleContract):
    values: list[float]

    @field_validator("values")
    @classmethod
    def validate_finite(cls, values: list[float]) -> list[float]:
        if not values:
            raise ValueError("values must contain at least one sample")
        if any(value != value or value in (float("inf"), float("-inf")) for value in values):
            raise ValueError("all samples must be finite")
        return values


class MultiChainRequest(SampleContract):
    chains: list[list[float]]
    chain_ids: list[str] | None = Field(default=None, max_length=10_000)
    require_equal_length: bool = True

    @field_validator("chains")
    @classmethod
    def validate_chains(cls, chains: list[list[float]]) -> list[list[float]]:
        if not chains:
            raise ValueError("chains must contain at least one chain")
        for chain in chains:
            if not chain:
                raise ValueError("each chain must contain at least one sample")
            if any(value != value or value in (float("inf"), float("-inf")) for value in chain):
                raise ValueError("all samples must be finite")
        return chains

    @model_validator(mode="after")
    def validate_chain_ids(self) -> MultiChainRequest:
        if self.chain_ids is not None:
            if len(self.chain_ids) != len(self.chains):
                raise ValueError("chain_ids length must match chains length")
            if len(set(self.chain_ids)) != len(self.chain_ids):
                raise ValueError("chain_ids must be unique")
        return self


class Diagnostic(BaseModel):
    code: str
    level: DiagnosticLevel
    message: str
    scope: str = "general"
    uncertainty: bool = False


class ACFResult(BaseModel):
    chain_id: str
    n: int
    mean: float
    max_lag: int
    lags: list[int]
    acf: list[float | None]
    constant: bool
    method: str


class AutocovSummary(BaseModel):
    lag_cutoff_rule: str
    reported_max_lag: int
    ess_scanned_lags: int
    centered: bool
    denominator: str


class ESSResult(BaseModel):
    kind: Literal["mean", "split_mean", "bulk"]
    ess: float | None
    total_draws: int
    relative_ess: float | None
    integrated_time: float | None
    cutoff_lag: int | None
    truncation_rule: str
    state: Literal["estimated", "undefined", "not_estimated"]


class VarianceComponents(BaseModel):
    n_chains: int
    draws_per_chain: int | None
    equal_length: bool
    within_chain_variance: float | None
    between_chain_variance: float | None
    marginal_variance: float | None
    degrees_of_freedom_between: int | None
    degrees_of_freedom_within: int | None
    formula: str


class RhatResult(BaseModel):
    kind: Literal["classic", "split", "rank_normalized"]
    rhat: float | None
    state: Literal["estimated", "undefined", "not_estimated"]
    reason: str | None = None
    threshold: float


class DiagnosisResult(BaseModel):
    request_id: str
    status: DiagnosticStatus
    n_chains: int
    total_draws: int
    draws: list[int]
    autocovariance: AutocovSummary
    acf: list[ACFResult]
    ess: list[ESSResult]
    variance_components: VarianceComponents | None = None
    rhat: list[RhatResult] = Field(default_factory=list)
    diagnostics: list[Diagnostic]
    limitations: list[str]
    steps: list[str]
    versions: dict[str, str]


class SingleChainResult(DiagnosisResult):
    pass


class MultiChainResult(DiagnosisResult):
    chain_ids: list[str]
