"""Pydantic API contracts."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

InputNumber = int | float | str


class ObjectiveSpec(BaseModel):
    name: str | None = None
    direction: Literal["min", "max"]


class NumericSettings(BaseModel):
    mode: Literal["exact", "quantized"] = "exact"
    tolerance: InputNumber | None = None


class CandidateInput(BaseModel):
    id: str | None = None
    objectives: list[InputNumber | None] | None = None
    payload: dict[str, Any] | None = None


class CandidateBatch(BaseModel):
    candidates: list[CandidateInput] | None = None


class CreateProblemRequest(BaseModel):
    problem_id: str | None = Field(default=None, min_length=1)
    objectives: list[ObjectiveSpec] | None = None
    numeric_settings: NumericSettings = Field(default_factory=NumericSettings)
    candidates: list[CandidateInput] = Field(default_factory=list)


class CandidateView(BaseModel):
    id: str
    objectives: list[InputNumber]
    payload: dict[str, Any] | None
    rank: int
    duplicate_group: str
    dominates: list[str]
    dominated_by: list[str]


class LayerView(BaseModel):
    rank: int
    candidate_ids: list[str]


class NumericPolicyView(BaseModel):
    mode: Literal["exact", "quantized"]
    tolerance: str | None
    exact_by_default: bool


class ProblemView(BaseModel):
    problem_id: str
    objective_count: int
    objective_names: list[str]
    directions: list[Literal["min", "max"]]
    numeric_policy: NumericPolicyView
    candidate_count: int
    layers: list[LayerView]
    candidates: list[CandidateView]


class CheckView(BaseModel):
    name: str
    passed: bool
    details: str


class EvidenceView(BaseModel):
    algorithm_version: str
    operation: str
    processing_stage: str
    steps: list[str]
    pair_comparisons: int
    edges_added: int
    edges_removed: int
    affected_candidate_ids: list[str]
    duplicate_groups: list[list[str]]
    checks: list[CheckView]
    uncertainty: list[str]


DataT = TypeVar("DataT")


class ServiceResponse(BaseModel, Generic[DataT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    version: str
    stage: str
    data: DataT
    evidence: EvidenceView | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


class ProblemData(BaseModel):
    state: ProblemView


class CandidateOperationData(BaseModel):
    candidate_id: str
    state: ProblemView
