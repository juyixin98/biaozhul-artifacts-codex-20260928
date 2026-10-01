"""Pydantic request and response contracts."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Direction(str, enum.Enum):
    minimize = "minimize"
    maximize = "maximize"


class VarType(str, enum.Enum):
    continuous = "continuous"
    integer = "integer"
    binary = "binary"


class Sense(str, enum.Enum):
    le = "<="
    ge = ">="
    eq = "=="


class VariableSpec(BaseModel):
    name: str = Field(min_length=1)
    type: VarType = VarType.continuous
    lower: float | None = None
    upper: float | None = None

    @field_validator("lower", "upper")
    @classmethod
    def finite_if_present(cls, value: float | None) -> float | None:
        if value is not None and (value != value or value in (float("inf"), float("-inf"))):
            raise ValueError("bounds must be finite")
        return value


class ConstraintSpec(BaseModel):
    name: str | None = None
    terms: dict[str, float]
    sense: Sense
    rhs: float

    @field_validator("terms")
    @classmethod
    def terms_not_empty(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("constraint must contain at least one term")
        return value

    @field_validator("rhs")
    @classmethod
    def finite_rhs(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("rhs must be finite")
        return value


class LPProblem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "problem"
    sense: Direction = Direction.minimize
    objective: dict[str, float]
    variables: list[VariableSpec]
    constraints: list[ConstraintSpec] = Field(default_factory=list)

    @field_validator("objective")
    @classmethod
    def objective_not_empty(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("objective must contain at least one variable")
        return value


class SolveOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None = Field(default=None, min_length=1, max_length=200)
    integer_tolerance: float | None = Field(default=None, ge=0.0)
    constraint_tolerance: float | None = Field(default=None, ge=0.0)
    lp_tolerance: float | None = Field(default=None, ge=0.0)
    time_limit_seconds: float | None = Field(default=None, ge=0.0)
    node_limit: int | None = Field(default=None, ge=1)
    branch_variable: str = "most_fractional"

    @field_validator("branch_variable")
    @classmethod
    def supported_strategy(cls, value: str) -> str:
        if value not in {"most_fractional", "first_fractional"}:
            raise ValueError("branch_variable must be 'most_fractional' or 'first_fractional'")
        return value


class Event(BaseModel):
    sequence: int
    run_id: str
    event: str
    reason: str
    node_id: int | None = None
    depth: int | None = None
    node_bound: float | None = None
    incumbent_value: float | None = None
    global_bound: float | None = None
    branch_variable: str | None = None
    branch_value: float | None = None
    child_bound: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "node_bound",
        "incumbent_value",
        "global_bound",
        "branch_value",
        "child_bound",
        mode="before",
    )
    @classmethod
    def non_finite_float_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and value in {"Infinity", "-Infinity", "NaN"}:
            return None
        if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
            return None
        return value


class NodeState(BaseModel):
    node_id: int
    parent_id: int | None
    depth: int
    branch_constraints: list[dict[str, Any]]
    bounds: dict[str, dict[str, float | None]]
    relaxation_status: str | None
    relaxation_objective: float | None
    bound: float | None


class SolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem: LPProblem
    options: SolveOptions = Field(default_factory=SolveOptions)


class SolveResponse(BaseModel):
    run_id: str
    status: str
    termination_reason: str
    sense: Direction
    objective_value: float | None
    best_bound: float | None
    absolute_gap: float | None
    relative_gap: float | None
    solution: dict[str, float] | None
    integer_infeasible_candidate_rejected: bool
    nodes_solved: int
    nodes_open: int
    events_omitted: int
    events: list[Event]
    open_nodes: list[NodeState]
    diagnostics: dict[str, Any]
    journal_path: str | None
