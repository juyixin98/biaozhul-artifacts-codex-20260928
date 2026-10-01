from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import numpy as np


SolveStatus = Literal["optimal", "infeasible", "failed", "uncertain"]


@dataclass
class SolverEvent:
    iteration: int
    phase: str
    type: str
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeasibilityResult:
    feasible: bool
    x: np.ndarray | None
    status: SolveStatus
    reason: str
    events: list[SolverEvent] = field(default_factory=list)
    max_primal_violation: float | None = None
    reference_solver: str = "scipy.optimize.linprog(method='highs')"
    reference_message: str = ""


@dataclass
class ActiveSetResult:
    x: np.ndarray
    status: SolveStatus
    reason: str
    events: list[SolverEvent]
    iterations: int
    active_inequality_indices: list[int]
    active_inequality_names: list[str]
    equality_basis_indices: list[int]
    dependent_equality_indices: list[int]
    warnings: list[str] = field(default_factory=list)
