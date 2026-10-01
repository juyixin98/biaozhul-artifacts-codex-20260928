"""Pydantic response schemas and request body types."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field


class ObjectiveSchema(BaseModel):
    name: str
    direction: Literal["min", "max"]


class CandidateInputSchema(BaseModel):
    id: str = Field(min_length=1)
    values: dict[str, float]


class CreateDatasetSchema(BaseModel):
    dataset_id: str | None = None
    objectives: list[ObjectiveSchema]
    candidates: list[CandidateInputSchema]
    tolerance: float | None = 0.0


class InsertCandidatesSchema(BaseModel):
    candidates: list[CandidateInputSchema] = Field(min_length=1)


class SortSchema(BaseModel):
    objectives: list[ObjectiveSchema]
    candidates: list[CandidateInputSchema]
    tolerance: float | None = 0.0


class CandidateOut(BaseModel):
    id: str
    values: dict[str, float]
    rank: int


class RankChangeOut(BaseModel):
    id: str
    from_rank: int
    to_rank: int


class EvidenceOut(BaseModel):
    comparisons_performed: int
    pairwise_dominance_edges: int
    duplicate_groups: list[list[str]]
    independent_reference_layers: list[list[str]]
    independent_reference_matched: bool
    layer_non_dominated: bool
    transitivity_preserved: bool
    ordering_predicate: str
    tolerance_applied_to_ordering: bool


class DiagnosticsOut(BaseModel):
    tolerance: float
    tolerance_role: str
    constant_objectives: list[str]
    near_equal_pairs: list[dict[str, Any]]
    duplicate_vector_count: int
    uncertainties: list[dict[str, Any]]


class DatasetOut(BaseModel):
    dataset_id: str
    api_version: str
    algorithm_version: str
    kernel_version: int
    objectives: list[ObjectiveSchema]
    tolerance: float
    candidates: list[CandidateOut]
    layers: list[list[str]]
    evidence: EvidenceOut
    diagnostics: DiagnosticsOut
    processing: list[dict[str, Any]]


class MutationOut(DatasetOut):
    operation: str
    inserted_ids: list[str] | None = None
    deleted_id: str | None = None
    affected_ids: list[str] | None = None
    rank_changes: list[RankChangeOut]


class HealthOut(BaseModel):
    status: str
    api_version: str
    algorithm_version: str
    location: str


ResultT = TypeVar("ResultT")


class Envelope(BaseModel, Generic[ResultT]):
    request_id: str
    status: Literal["ok", "error"]
    result: ResultT
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class ErrorOut(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None
