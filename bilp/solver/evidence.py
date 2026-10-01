"""Evidence records and numerical diagnostics.

Every processed node yields a NodeEvidence with the branch constraints, the
materialized bounds, the relaxation outcome, the decision taken and a
human-readable reason carrying the numbers behind the decision. This module
also hosts the feasibility/integrality diagnostics applied to incumbent
candidates and the optimality-gap computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..model import CanonicalProblem

# Node decisions.
PRUNED_BY_BOUND = "pruned_by_bound"
PRUNED_INFEASIBLE = "pruned_infeasible"
PRUNED_INTEGRAL = "pruned_integral"
BRANCHED = "branched"
UNBOUNDED = "unbounded"


def _f(v: float | None) -> float | None:
    """JSON-safe float: +/-inf become null (position implies the sign)."""
    if v is None:
        return None
    v = float(v)
    if np.isinf(v):
        return None
    return v


@dataclass
class NodeEvidence:
    node_id: int
    parent_id: int | None
    depth: int
    branch_constraints: list[dict]
    lb: list[float | None]
    ub: list[float | None]
    relaxation_status: str                    # optimal | infeasible | not_solved
    relaxation_objective: float | None        # original optimization sense
    bound_used: float | None                  # bound the decision relied on
    incumbent_at_node: float | None           # incumbent before this node
    decision: str
    reason: str
    branch_var: int | None = None
    branch_value: float | None = None
    candidate_rejected: bool = False
    candidate_violation: float | None = None

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "branch_constraints": self.branch_constraints,
            "lb": [_f(v) for v in self.lb],
            "ub": [_f(v) for v in self.ub],
            "relaxation_status": self.relaxation_status,
            "relaxation_objective": _f(self.relaxation_objective),
            "bound_used": _f(self.bound_used),
            "incumbent_at_node": _f(self.incumbent_at_node),
            "decision": self.decision,
            "reason": self.reason,
            "branch_var": self.branch_var,
            "branch_value": _f(self.branch_value),
            "candidate_rejected": self.candidate_rejected,
            "candidate_violation": _f(self.candidate_violation),
        }


@dataclass
class EvidenceRecorder:
    """Keeps per-node evidence up to a cap; always counts every node."""

    cap: int
    records: list[NodeEvidence] = field(default_factory=list)
    total_recorded: int = 0
    truncated: bool = False

    def record(self, ev: NodeEvidence) -> None:
        self.total_recorded += 1
        if len(self.records) < self.cap:
            self.records.append(ev)
        else:
            self.truncated = True

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self.records]


def max_constraint_violation(problem: CanonicalProblem, x: np.ndarray) -> float:
    """Largest violation of any constraint (A_ub, A_eq, bounds) at x."""
    viol = 0.0
    if problem.A_ub.shape[0]:
        viol = max(viol, float(np.max(problem.A_ub @ x - problem.b_ub)))
    if problem.A_eq.shape[0]:
        viol = max(viol, float(np.max(np.abs(problem.A_eq @ x - problem.b_eq))))
    viol = max(viol, float(np.max(problem.lb - x)), float(np.max(x - problem.ub)))
    return max(viol, 0.0)


def max_integrality_violation(problem: CanonicalProblem, x: np.ndarray) -> float:
    if problem.integer_idx.size == 0:
        return 0.0
    fracs = np.abs(x[problem.integer_idx] - np.round(x[problem.integer_idx]))
    return float(np.max(fracs))


def compute_gaps(maximize: bool, incumbent: float | None,
                 best_bound: float | None) -> tuple[float | None, float | None]:
    """(absolute_gap, relative_gap); relative gap uses max(1, |incumbent|)."""
    if incumbent is None:
        return None, None
    if best_bound is None:
        return 0.0, 0.0
    if maximize:
        abs_gap = best_bound - incumbent
    else:
        abs_gap = incumbent - best_bound
    abs_gap = max(abs_gap, 0.0)  # numerical noise guard
    rel_gap = abs_gap / max(1.0, abs(incumbent))
    return abs_gap, rel_gap
