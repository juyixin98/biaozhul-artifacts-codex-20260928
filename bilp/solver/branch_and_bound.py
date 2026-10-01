"""Branch-and-bound kernel.

LP relaxations are solved by SciPy's HiGHS backend (scipy.optimize.linprog).
The search maintains an incumbent (best feasible integer solution found so
far) and a frontier of open nodes, each carrying a valid relaxation bound
(a parent's bound is a valid bound for its children).

Pruning is direction-aware: for minimization a node is pruned when its
relaxation bound >= incumbent - tol; for maximization when bound <=
incumbent + tol. Integer candidates are rounded and then re-verified
against every constraint with a strict tolerance, so the integer tolerance
can never smuggle an infeasible point into the incumbent.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import linprog

from ..config import SolverConfig
from ..errors import ComputationFailureError
from ..model import CanonicalProblem
from .evidence import (
    BRANCHED,
    PRUNED_BY_BOUND,
    PRUNED_INFEASIBLE,
    PRUNED_INTEGRAL,
    UNBOUNDED,
    EvidenceRecorder,
    NodeEvidence,
    compute_gaps,
    max_constraint_violation,
    max_integrality_violation,
)
from .node import BranchConstraint, Node, NodeArchive


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"                        # proven optimal
    INFEASIBLE = "infeasible"                  # root LP relaxation infeasible
    INTEGER_INFEASIBLE = "integer_infeasible"  # relaxation feasible, no integer point
    FEASIBLE = "feasible"                      # budget stop with an incumbent
    UNKNOWN = "unknown"                        # budget stop without an incumbent
    UNBOUNDED_RELAXATION = "unbounded_relaxation"  # root relaxation unbounded


@dataclass
class SolveResult:
    run_id: str
    status: SolveStatus
    proven_optimal: bool
    objective: float | None
    solution: list[float] | None
    optimality_gap: float | None
    absolute_gap: float | None
    best_bound: float | None
    termination_reason: str
    diagnostics: dict
    evidence: list[dict]
    evidence_truncated: bool
    archive: NodeArchive  # kept for node restoration; not serialized

    def to_dict(self, include_evidence: bool = False) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "proven_optimal": self.proven_optimal,
            "objective": self.objective,
            "solution": self.solution,
            "optimality_gap": self.optimality_gap,
            "absolute_gap": self.absolute_gap,
            "best_bound": self.best_bound,
            "termination_reason": self.termination_reason,
            "diagnostics": self.diagnostics,
            "evidence": self.evidence if include_evidence else None,
            "evidence_truncated": self.evidence_truncated,
        }


class BranchAndBoundSolver:
    def __init__(self, problem: CanonicalProblem, config: SolverConfig, run_id: str = ""):
        self.problem = problem
        self.config = config
        self.run_id = run_id

    def _solve_relaxation(self, lb: np.ndarray, ub: np.ndarray):
        """Solve the LP relaxation; returns (status, x, objective in original sense)."""
        p = self.problem
        c_lp = -p.c if p.maximize else p.c
        res = linprog(
            c_lp,
            A_ub=p.A_ub if p.A_ub.shape[0] else None,
            b_ub=p.b_ub if p.b_ub.shape[0] else None,
            A_eq=p.A_eq if p.A_eq.shape[0] else None,
            b_eq=p.b_eq if p.b_eq.shape[0] else None,
            bounds=[(float(lo), float(hi)) for lo, hi in zip(lb, ub)],
            method=self.config.lp_method,
        )
        if res.status == 0:
            x = np.asarray(res.x, dtype=float)
            if x is None or not np.all(np.isfinite(x)):
                raise ComputationFailureError(
                    "LP relaxation returned a non-finite solution",
                    {"solver_status": int(res.status)},
                )
            return "optimal", x, float(p.c @ x)
        if res.status == 2:
            return "infeasible", None, None
        if res.status == 3:
            return "unbounded", None, None
        raise ComputationFailureError(
            f"LP relaxation solver failed (status {res.status}: {res.message})",
            {"solver_status": int(res.status), "solver_message": str(res.message)},
        )

    def _bound_prunes(self, bound: float, incumbent: float) -> bool:
        tol = self.config.prune_tol
        if self.problem.maximize:
            return bound <= incumbent + tol
        return bound >= incumbent - tol

    def _is_better(self, obj: float, incumbent: float) -> bool:
        tol = self.config.prune_tol
        if self.problem.maximize:
            return obj > incumbent + tol
        return obj < incumbent - tol

    def solve(self) -> SolveResult:
        t0 = time.monotonic()
        p, cfg = self.problem, self.config
        archive = NodeArchive(p.lb, p.ub, bounds_retention=cfg.max_evidence_nodes)
        recorder = EvidenceRecorder(cap=cfg.max_evidence_nodes)
        int_idx = p.integer_idx
        sign = -1.0 if p.maximize else 1.0  # heap key: original-sense bound to min-space

        incumbent_x: np.ndarray | None = None
        incumbent_obj: float | None = None
        nodes_processed = 0
        lp_solves = 0
        n_bound = n_infeas = n_integral = n_rejected = 0
        root_relaxation_infeasible = False
        unbounded_root = False
        termination: str | None = None

        frontier: list[tuple[float, int, Node, float | None]] = []
        root = Node(node_id=0, parent_id=None, depth=0, constraints=())
        heapq.heappush(frontier, (0.0, root.node_id, root, None))
        next_id = 1

        def budget_stop() -> str | None:
            # The root relaxation is always solved, so even an immediate
            # budget stop still yields a valid root bound.
            if nodes_processed == 0:
                return None
            if nodes_processed >= cfg.max_nodes:
                return "node_budget_exhausted"
            if time.monotonic() - t0 >= cfg.time_limit_s:
                return "time_budget_exhausted"
            return None

        while frontier:
            stop = budget_stop()
            if stop is not None:
                termination = stop
                break
            _, _, node, _ = heapq.heappop(frontier)
            lb, ub = archive.register(node)
            ev_base = dict(
                node_id=node.node_id,
                parent_id=node.parent_id,
                depth=node.depth,
                branch_constraints=[bc.to_dict() for bc in node.constraints],
                lb=lb.tolist(),
                ub=ub.tolist(),
                incumbent_at_node=incumbent_obj,
            )

            if np.any(lb > ub):
                n_infeas += 1
                nodes_processed += 1
                recorder.record(NodeEvidence(
                    **ev_base,
                    relaxation_status="infeasible",
                    relaxation_objective=None,
                    bound_used=None,
                    decision=PRUNED_INFEASIBLE,
                    reason="branch constraints imply lb > ub for some variable; node is empty",
                ))
                continue

            status, x, obj = self._solve_relaxation(lb, ub)
            lp_solves += 1
            nodes_processed += 1

            if status == "infeasible":
                if node.parent_id is None:
                    root_relaxation_infeasible = True
                n_infeas += 1
                recorder.record(NodeEvidence(
                    **ev_base,
                    relaxation_status="infeasible",
                    relaxation_objective=None,
                    bound_used=None,
                    decision=PRUNED_INFEASIBLE,
                    reason="LP relaxation is infeasible at this node",
                ))
                continue

            if status == "unbounded":
                if node.parent_id is None:
                    unbounded_root = True
                    termination = "root_relaxation_unbounded"
                    recorder.record(NodeEvidence(
                        **ev_base,
                        relaxation_status="unbounded",
                        relaxation_objective=None,
                        bound_used=None,
                        decision=UNBOUNDED,
                        reason="root LP relaxation is unbounded; the service cannot "
                               "certify an optimum (add finite bounds on continuous "
                               "variables or objective-bounding constraints)",
                    ))
                    break
                raise ComputationFailureError(
                    "child relaxation unbounded although the root relaxation was "
                    "bounded; this indicates a numerical inconsistency",
                    {"node_id": node.node_id},
                )

            bound = obj  # original optimization sense

            if incumbent_obj is not None and self._bound_prunes(bound, incumbent_obj):
                n_bound += 1
                direction = "maximization: bound <= incumbent + tol" if p.maximize \
                    else "minimization: bound >= incumbent - tol"
                recorder.record(NodeEvidence(
                    **ev_base,
                    relaxation_status="optimal",
                    relaxation_objective=bound,
                    bound_used=bound,
                    decision=PRUNED_BY_BOUND,
                    reason=f"relaxation bound {bound:.6g} cannot improve incumbent "
                           f"{incumbent_obj:.6g} ({direction}, prune_tol={cfg.prune_tol:.1g})",
                ))
                continue

            if int_idx.size:
                fracs = np.abs(x[int_idx] - np.round(x[int_idx]))
                max_frac = float(np.max(fracs))
            else:
                fracs = np.zeros(0)
                max_frac = 0.0

            candidate_rejected = False
            candidate_violation: float | None = None
            reject_reason = ""
            if max_frac <= cfg.int_tol:
                candidate = x.copy()
                if int_idx.size:
                    candidate[int_idx] = np.round(x[int_idx])
                candidate_violation = max_constraint_violation(p, candidate)
                allowed = cfg.candidate_feas_tol * p.tol_scale
                if candidate_violation <= allowed:
                    cand_obj = float(p.c @ candidate)
                    improved = incumbent_obj is None or self._is_better(cand_obj, incumbent_obj)
                    if improved:
                        incumbent_obj = cand_obj
                        incumbent_x = candidate
                    n_integral += 1
                    recorder.record(NodeEvidence(
                        **ev_base,
                        relaxation_status="optimal",
                        relaxation_objective=bound,
                        bound_used=bound,
                        decision=PRUNED_INTEGRAL,
                        reason=f"integer-feasible candidate accepted (constraint "
                               f"violation {candidate_violation:.3g} <= {allowed:.3g}); "
                               f"incumbent {'updated to ' + format(cand_obj, '.6g') if improved else 'unchanged'}",
                        candidate_violation=candidate_violation,
                    ))
                    continue
                # Rounded candidate is infeasible: never accept it; branch instead.
                if max_frac == 0.0:
                    raise ComputationFailureError(
                        "LP solution is exactly integral but violates constraints "
                        "beyond the candidate tolerance",
                        {"node_id": node.node_id, "violation": candidate_violation,
                         "allowed": allowed},
                    )
                n_rejected += 1
                candidate_rejected = True
                reject_reason = (
                    f"rounded integer candidate violates constraints by "
                    f"{candidate_violation:.3g} > {allowed:.3g}; branching instead "
                    "of accepting an illegal solution"
                )

            # Branch on the most fractional integer variable.
            j = int(int_idx[int(np.argmax(fracs))])
            v = float(x[j])
            lo_value = float(math.floor(v))
            hi_value = float(math.ceil(v))
            lo = Node(next_id, node.node_id, node.depth + 1,
                      node.constraints + (BranchConstraint(j, "<=", lo_value),))
            next_id += 1
            hi = Node(next_id, node.node_id, node.depth + 1,
                      node.constraints + (BranchConstraint(j, ">=", hi_value),))
            next_id += 1
            key = sign * bound
            heapq.heappush(frontier, (key, lo.node_id, lo, bound))
            heapq.heappush(frontier, (key, hi.node_id, hi, bound))
            branch_reason = reject_reason or (
                f"x[{j}]={v:.6g} is fractional (max fractionality {max_frac:.3g}); "
                f"branch x[{j}] <= {lo_value:.6g} and x[{j}] >= {hi_value:.6g}"
            )
            recorder.record(NodeEvidence(
                **ev_base,
                relaxation_status="optimal",
                relaxation_objective=bound,
                bound_used=bound,
                decision=BRANCHED,
                reason=branch_reason,
                branch_var=j,
                branch_value=v,
                candidate_rejected=candidate_rejected,
                candidate_violation=candidate_violation,
            ))

        # Resolve final status.
        frontier_empty = not frontier
        if termination is None:
            termination = "all_nodes_pruned"

        if unbounded_root:
            final_status = SolveStatus.UNBOUNDED_RELAXATION
        elif frontier_empty:
            if root_relaxation_infeasible:
                final_status = SolveStatus.INFEASIBLE
                termination = "root_relaxation_infeasible"
            elif incumbent_obj is not None:
                final_status = SolveStatus.OPTIMAL
            else:
                final_status = SolveStatus.INTEGER_INFEASIBLE
        else:
            final_status = SolveStatus.FEASIBLE if incumbent_obj is not None else SolveStatus.UNKNOWN

        # Best bound still available to the search.
        if final_status == SolveStatus.OPTIMAL:
            best_bound = incumbent_obj
        elif final_status in (SolveStatus.FEASIBLE, SolveStatus.UNKNOWN):
            estimates = [est for _, _, _, est in frontier if est is not None]
            if estimates:
                best_bound = max(estimates) if p.maximize else min(estimates)
            else:
                best_bound = None
        else:
            best_bound = None

        absolute_gap, optimality_gap = compute_gaps(p.maximize, incumbent_obj, best_bound)

        diagnostics = {
            "nodes_processed": nodes_processed,
            "lp_solves": lp_solves,
            "nodes_pruned_by_bound": n_bound,
            "nodes_pruned_infeasible": n_infeas,
            "nodes_pruned_integral": n_integral,
            "candidate_rejections": n_rejected,
            "elapsed_s": round(time.monotonic() - t0, 6),
            "incumbent_max_constraint_violation": (
                max_constraint_violation(p, incumbent_x) if incumbent_x is not None else None
            ),
            "incumbent_max_integrality_violation": (
                max_integrality_violation(p, incumbent_x) if incumbent_x is not None else None
            ),
        }

        return SolveResult(
            run_id=self.run_id,
            status=final_status,
            proven_optimal=(final_status == SolveStatus.OPTIMAL),
            objective=incumbent_obj,
            solution=incumbent_x.tolist() if incumbent_x is not None else None,
            optimality_gap=optimality_gap,
            absolute_gap=absolute_gap,
            best_bound=best_bound,
            termination_reason=termination,
            diagnostics=diagnostics,
            evidence=recorder.to_list(),
            evidence_truncated=recorder.truncated,
            archive=archive,
        )
