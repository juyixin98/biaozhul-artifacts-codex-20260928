"""Pydantic request/response schemas for the HTTP API.

Semantic validation (dimensions, finiteness, boundedness of integer
variables) lives in ``bilp.model`` so it can be reused off-HTTP; these
models only enforce shape and types.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProblemSpec(BaseModel):
    """MILP in dense form.

    minimize/maximize  c @ x
    subject to         A_ub @ x <= b_ub
                       A_eq @ x == b_eq
                       bounds[i][0] <= x[i] <= bounds[i][1]   (null = unbounded)
                       x[i] integer where integrality[i] == 1
    """

    model_config = ConfigDict(extra="forbid")

    sense: Literal["minimize", "maximize"] = "minimize"
    c: list[float]
    integrality: list[int] | None = None          # default: all integer
    bounds: list[tuple[float | None, float | None]] | None = None  # default: [0, inf)
    A_ub: list[list[float]] | None = None
    b_ub: list[float] | None = None
    A_eq: list[list[float]] | None = None
    b_eq: list[float] | None = None


class SolveOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_nodes: int | None = None
    time_limit_s: float | None = None
    int_tol: float | None = None
    include_evidence: bool = False


class SolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem: ProblemSpec
    options: SolveOptions = Field(default_factory=SolveOptions)
    client_request_id: str | None = None


class SolveResponse(BaseModel):
    run_id: str
    status: str
    proven_optimal: bool
    objective: float | None
    solution: list[float] | None
    optimality_gap: float | None
    absolute_gap: float | None
    best_bound: float | None
    termination_reason: str
    diagnostics: dict[str, Any]
    evidence: list[dict[str, Any]] | None = None
    evidence_truncated: bool
    idempotent_replay: bool = False
