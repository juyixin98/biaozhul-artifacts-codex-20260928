"""Branch-and-bound kernel using SciPy HiGHS for LP relaxations."""

from __future__ import annotations

import heapq
import math
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linprog

from .config import Settings
from .diagnostics import rounded_integer_candidate, verify_candidate
from .errors import ComputationError
from .journal import RunJournal, new_run_id
from .model import ParsedProblem
from .schemas import LPProblem, SolveOptions


@dataclass
class Node:
    node_id: int
    parent_id: int | None
    depth: int
    bounds: dict[int, tuple[float, float]]
    branch_constraints: list[dict[str, Any]]
    bound: float
    relaxed: bool = False
    relaxation_status: str | None = None
    relaxation_objective: float | None = None
    x: np.ndarray | None = None


def heap_key(node: Node) -> tuple[int, float, int]:
    return (-node.depth, node.bound, -node.node_id)


@dataclass
class SolveResult:
    run_id: str
    status: str
    termination_reason: str
    sense: str
    internal_best_bound: float | None
    internal_incumbent_value: float | None
    incumbent: np.ndarray | None
    nodes_solved: int
    events: list[dict[str, Any]]
    open_nodes: list[Node]
    diagnostics: dict[str, Any]
    journal_path: str
    rejected_integer_candidate: bool = False

    @property
    def objective_value(self) -> float | None:
        if self.internal_incumbent_value is None:
            return None
        return -self.internal_incumbent_value if self.sense == "maximize" else self.internal_incumbent_value

    @property
    def best_bound(self) -> float | None:
        if self.internal_best_bound is None:
            return None
        return -self.internal_best_bound if self.sense == "maximize" else self.internal_best_bound

    @property
    def absolute_gap(self) -> float | None:
        if self.internal_incumbent_value is None or self.internal_best_bound is None:
            return None
        return float(abs(self.internal_incumbent_value - self.internal_best_bound))


def _resolve_options(options: SolveOptions, settings: Settings) -> tuple[float, float, float, int]:
    integer_tolerance = settings.default_integer_tolerance if options.integer_tolerance is None else options.integer_tolerance
    constraint_tolerance = settings.default_constraint_tolerance if options.constraint_tolerance is None else options.constraint_tolerance
    lp_tolerance = settings.default_lp_tolerance if options.lp_tolerance is None else options.lp_tolerance
    node_limit = options.node_limit or settings.default_node_limit
    return integer_tolerance, constraint_tolerance, lp_tolerance, node_limit


def _bounds_feasible(lower: np.ndarray, upper: np.ndarray) -> tuple[bool, int]:
    for i in range(lower.size):
        if lower[i] > upper[i]:
            return False, i
    return True, -1


def _serialise_bounds(problem: ParsedProblem, bounds: dict[int, tuple[float, float]]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for index, (lower, upper) in sorted(bounds.items()):
        result[problem.names[index]] = {
            "lower": _clean_zero(lower) if math.isfinite(lower) else None,
            "upper": _clean_zero(upper) if math.isfinite(upper) else None,
        }
    return result


def _clean_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


def _json_number(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return _clean_zero(float(value))


def _node_state(node: Node, problem: ParsedProblem) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "depth": node.depth,
        "branch_constraints": node.branch_constraints,
        "bounds": _serialise_bounds(problem, node.bounds),
        "relaxation_status": node.relaxation_status,
        "relaxation_objective": node.relaxation_objective,
        "bound": node.bound,
    }


def solve_problem(
    problem: ParsedProblem,
    options: SolveOptions,
    settings: Settings,
    journal: RunJournal,
    time_limit_seconds: float,
) -> SolveResult:
    integer_tolerance, constraint_tolerance, lp_tolerance, node_limit = _resolve_options(options, settings)
    start = time.perf_counter()
    deadline = start + time_limit_seconds
    events: list[dict[str, Any]] = []

    def current_global_bound(open_heap: list[tuple[tuple[int, float, int], Node]]) -> float | None:
        bounds = [entry[1].bound for entry in open_heap if math.isfinite(entry[1].bound)]
        if not bounds:
            return None
        return float(min(bounds))

    def emit(
        event: str,
        reason: str,
        *,
        node: Node | None = None,
        incumbent_value: float | None = None,
        global_bound: float | None = None,
        **detail: Any,
    ) -> dict[str, Any]:
        payload = {
            "run_id": journal.run_id,
            "event": event,
            "reason": reason,
            "node_id": node.node_id if node else None,
            "depth": node.depth if node else None,
            "node_bound": _json_number(node.bound) if node else None,
            "incumbent_value": incumbent_value,
            "global_bound": global_bound,
            "branch_variable": detail.pop("branch_variable", None),
            "branch_value": _json_number(detail.pop("branch_value", None)),
            "child_bound": detail.pop("child_bound", None),
            "detail": detail,
        }
        recorded = journal.record(payload)
        events.append(recorded)
        return recorded

    root_lower, root_upper = problem.variable_bounds()
    feasible, bad_index = _bounds_feasible(root_lower, root_upper)
    root = Node(0, None, 0, {}, [], -math.inf)
    processed_nodes: dict[int, Node] = {}
    heap: list[tuple[tuple[int, float, int], Node]] = []
    incumbent: np.ndarray | None = None
    incumbent_value: float | None = None
    nodes_solved = 0
    next_node_id = 1
    rejected_integer_candidate = False
    final_reason = "optimal"
    last_prune_reason: str | None = None

    if not feasible:
        root.relaxation_status = "infeasible_before_lp"
        root.bound = math.inf
        processed_nodes[0] = root
        last_prune_reason = "empty_integer_domain"
        emit(
            "node_pruned",
            "empty_integer_domain",
            node=root,
            incumbent_value=incumbent_value,
            global_bound=None,
            variable=problem.names[bad_index],
            bounds=_serialise_bounds(problem, {bad_index: (float(root_lower[bad_index]), float(root_upper[bad_index]))}),
        )
        return SolveResult(
            run_id=journal.run_id,
            status="infeasible",
            termination_reason="empty_integer_domain",
            sense=problem.sense,
            internal_best_bound=None,
            internal_incumbent_value=None,
            incumbent=None,
            nodes_solved=0,
            events=events,
            open_nodes=[],
            diagnostics={"integer_tolerance": integer_tolerance, "constraint_tolerance": constraint_tolerance},
            journal_path=str(journal.path),
        )

    heapq.heappush(heap, (heap_key(root), root))

    while heap:
        if time.perf_counter() >= deadline:
            final_reason = "time_limit"
            break

        _, node = heapq.heappop(heap)
        lower, upper = problem.variable_bounds(node.bounds)
        feasible, bad_index = _bounds_feasible(lower, upper)
        if not feasible:
            node.relaxation_status = "infeasible_before_lp"
            node.bound = math.inf
            processed_nodes[node.node_id] = node
            last_prune_reason = "empty_integer_domain"
            emit(
                "node_pruned",
                "empty_integer_domain",
                node=node,
                incumbent_value=incumbent_value,
                global_bound=current_global_bound(heap),
                variable=problem.names[bad_index],
            )
            continue

        if incumbent_value is not None and node.bound >= incumbent_value:
            node.relaxation_status = "pruned_by_bound"
            processed_nodes[node.node_id] = node
            emit(
                "node_pruned",
                "bound",
                node=node,
                incumbent_value=incumbent_value,
                global_bound=current_global_bound(heap),
            )
            continue

        if nodes_solved >= node_limit:
            heapq.heappush(heap, (heap_key(node), node))
            final_reason = "node_limit"
            break

        try:
            lp_result = linprog(
                c=problem.objective,
                A_ub=problem.a_ub if problem.a_ub.shape[0] else None,
                b_ub=problem.b_ub if problem.b_ub.size else None,
                A_eq=problem.a_eq if problem.a_eq.shape[0] else None,
                b_eq=problem.b_eq if problem.b_eq.size else None,
            bounds=list(zip(lower, upper, strict=True)),
            method="highs",
            options={
                "primal_feasibility_tolerance": lp_tolerance,
                "dual_feasibility_tolerance": lp_tolerance,
                "time_limit": max(0.0, deadline - time.perf_counter()),
            },
            )
        except Exception as exc:
            raise ComputationError(f"LP solver raised {type(exc).__name__}: {exc}") from exc

        nodes_solved += 1

        if lp_result.status == 1:
            node.relaxation_status = "time_limited"
            node.relaxation_objective = None
            heapq.heappush(heap, (heap_key(node), node))
            final_reason = "time_limit"
            emit(
                "node_deferred",
                "lp_time_limit",
                node=node,
                incumbent_value=incumbent_value,
                global_bound=current_global_bound(heap),
                lp_message=lp_result.message,
            )
            break

        if lp_result.status == 2:
            node.relaxation_status = "infeasible"
            node.bound = math.inf
            processed_nodes[node.node_id] = node
            last_prune_reason = "lp_infeasible"
            emit(
                "node_pruned",
                "lp_infeasible",
                node=node,
                incumbent_value=incumbent_value,
                global_bound=current_global_bound(heap),
                lp_message=lp_result.message,
            )
            continue
        if lp_result.status == 3:
            node.relaxation_status = "unbounded"
            processed_nodes[node.node_id] = node
            final_reason = "lp_unbounded"
            emit(
                "solve_stopped",
                "lp_unbounded",
                node=node,
                incumbent_value=incumbent_value,
                global_bound=None,
                lp_message=lp_result.message,
            )
            break
        if (
            lp_result.status != 0
            or not lp_result.success
            or not np.isfinite(lp_result.fun)
            or lp_result.x is None
            or not np.all(np.isfinite(lp_result.x))
        ):
            raise ComputationError(f"LP solver failed with status {lp_result.status}: {lp_result.message}")

        node.relaxed = True
        node.relaxation_status = "optimal"
        node.relaxation_objective = float(lp_result.fun)
        node.bound = float(lp_result.fun)
        node.x = np.asarray(lp_result.x, dtype=float)
        processed_nodes[node.node_id] = node

        if incumbent_value is not None and node.bound >= incumbent_value:
            emit(
                "node_pruned",
                "bound",
                node=node,
                incumbent_value=incumbent_value,
                global_bound=current_global_bound(heap),
            )
            continue

        integer_indices = np.flatnonzero(problem.integer)
        integer_residuals = np.abs(node.x[problem.integer] - np.rint(node.x[problem.integer]))
        max_integer_residual = float(np.max(integer_residuals)) if integer_residuals.size else 0.0
        is_integral = max_integer_residual <= integer_tolerance

        emit(
            "lp_solved",
            "integral_relaxation" if is_integral else "fractional_relaxation",
            node=node,
            incumbent_value=incumbent_value,
            global_bound=current_global_bound(heap),
            max_integer_residual=max_integer_residual,
            integer_tolerance=integer_tolerance,
        )

        if is_integral:
            candidate = rounded_integer_candidate(node.x, problem.integer)
            check = verify_candidate(
                problem,
                candidate,
                integer_tolerance=integer_tolerance,
                constraint_tolerance=constraint_tolerance,
            )
            if not check.feasible:
                rejected_integer_candidate = True
                emit(
                    "candidate_rejected",
                    check.failure_category or "infeasible",
                    node=node,
                    incumbent_value=incumbent_value,
                    global_bound=current_global_bound(heap),
                    diagnostics={
                        "max_integer_residual": check.max_integer_residual,
                        "max_bound_violation": check.max_bound_violation,
                        "max_inequality_violation": check.max_inequality_violation,
                        "max_equality_residual": check.max_equality_residual,
                        "message": check.message,
                    },
                )
                continue

            candidate_value = check.internal_objective
            assert candidate_value is not None
            if incumbent_value is None or candidate_value < incumbent_value:
                incumbent = candidate
                incumbent_value = float(candidate_value)
                emit(
                    "incumbent_updated",
                    "validated_integer_candidate",
                    node=node,
                    incumbent_value=incumbent_value,
                    global_bound=current_global_bound(heap),
                    solution={problem.names[i]: _clean_zero(incumbent[i]) for i in range(problem.n)},
                )
            else:
                emit(
                    "node_pruned",
                    "not_better_than_incumbent",
                    node=node,
                    incumbent_value=incumbent_value,
                    global_bound=current_global_bound(heap),
                    candidate_value=float(candidate_value),
                )
            continue

        if options.branch_variable == "first_fractional":
            fractional_positions = np.flatnonzero(integer_residuals > integer_tolerance)
            local_index = int(fractional_positions[0])
        else:
            local_index = int(np.argmax(integer_residuals))
        branch_index = int(integer_indices[local_index])
        branch_value = float(node.x[branch_index])
        floor_value = float(np.floor(branch_value))
        ceil_value = floor_value + 1.0

        current_lower, current_upper = problem.variable_bounds(node.bounds)
        var_lower = float(current_lower[branch_index])
        var_upper = float(current_upper[branch_index])

        lower_bounds = dict(node.bounds)
        lower_bounds[branch_index] = (var_lower, min(var_upper, floor_value))
        upper_bounds = dict(node.bounds)
        upper_bounds[branch_index] = (max(var_lower, ceil_value), var_upper)

        lower_child = Node(
            node_id=next_node_id,
            parent_id=node.node_id,
            depth=node.depth + 1,
            bounds=lower_bounds,
            branch_constraints=node.branch_constraints
            + [
                {
                    "variable": problem.names[branch_index],
                    "operator": "<=",
                    "value": floor_value,
                    "parent_node": node.node_id,
                }
            ],
            bound=node.bound,
        )
        upper_child = Node(
            node_id=next_node_id + 1,
            parent_id=node.node_id,
            depth=node.depth + 1,
            bounds=upper_bounds,
            branch_constraints=node.branch_constraints
            + [
                {
                    "variable": problem.names[branch_index],
                    "operator": ">=",
                    "value": ceil_value,
                    "parent_node": node.node_id,
                }
            ],
            bound=node.bound,
        )
        next_node_id += 2

        # Equal-depth ties pop the larger node id, giving x >= ceil before x <= floor.
        heapq.heappush(heap, (heap_key(upper_child), upper_child))
        heapq.heappush(heap, (heap_key(lower_child), lower_child))
        emit(
            "branch",
            "fractional_integer_variable",
            node=node,
            incumbent_value=incumbent_value,
            global_bound=current_global_bound(heap),
            branch_variable=problem.names[branch_index],
            branch_value=branch_value,
            children=[_node_state(lower_child, problem), _node_state(upper_child, problem)],
        )

    open_nodes = [entry[1] for entry in heap]
    open_nodes.sort(key=lambda candidate: (candidate.bound, candidate.depth, candidate.node_id))
    finite_open_bounds = [node.bound for node in open_nodes if math.isfinite(node.bound)]
    internal_best_bound = min(finite_open_bounds) if finite_open_bounds else incumbent_value

    if final_reason in {"time_limit", "node_limit"}:
        status = "resource_exhausted"
    elif final_reason == "lp_unbounded":
        status = "unbounded" if not np.any(problem.integer) else "relaxation_unbounded"
        internal_best_bound = None
    elif incumbent is None:
        status = "infeasible"
        final_reason = last_prune_reason or "all_nodes_pruned"
        internal_best_bound = None
    else:
        status = "optimal"
        final_reason = "all_open_nodes_pruned"

    diagnostics = {
        "integer_tolerance": integer_tolerance,
        "constraint_tolerance": constraint_tolerance,
        "lp_tolerance": lp_tolerance,
        "node_limit": node_limit,
        "time_limit_seconds": time_limit_seconds,
        "elapsed_seconds": time.perf_counter() - start,
        "processed_node_count": len(processed_nodes),
        "rejected_integer_candidate": rejected_integer_candidate,
        "max_open_node_bound": max((node.bound for node in open_nodes), default=None),
        "min_open_node_bound": min((node.bound for node in open_nodes), default=None),
    }

    return SolveResult(
        run_id=journal.run_id,
        status=status,
        termination_reason=final_reason,
        sense=problem.sense,
        internal_best_bound=internal_best_bound,
        internal_incumbent_value=incumbent_value,
        incumbent=incumbent,
        nodes_solved=nodes_solved,
        events=events,
        open_nodes=open_nodes,
        diagnostics=diagnostics,
        journal_path=str(journal.path),
        rejected_integer_candidate=rejected_integer_candidate,
    )


def solve(
    problem: LPProblem,
    options: SolveOptions | None = None,
    *,
    settings: Settings | None = None,
) -> SolveResult:
    from .model import parse_problem

    settings = settings or Settings.from_env()
    options = options or SolveOptions()
    parsed = parse_problem(problem)

    time_limit = (
        options.time_limit_seconds
        if options.time_limit_seconds is not None
        else settings.default_time_limit_seconds
    )
    run_id = options.run_id or new_run_id()
    journal = RunJournal(
        run_id=run_id,
        journal_dir=settings.journal_dir,
        problem=problem.model_dump(mode="json"),
        options=options.model_dump(mode="json"),
    )
    try:
        result = solve_problem(parsed, options, settings, journal, time_limit)
        journal.complete(
            {
                "status": result.status,
                "termination_reason": result.termination_reason,
                "objective_value": result.objective_value,
                "best_bound": result.best_bound,
                "absolute_gap": result.absolute_gap,
                "nodes_solved": result.nodes_solved,
                "nodes_open": len(result.open_nodes),
            }
        )
        return result
    except Exception:
        journal.record(
            {
                "event": "solve_failed",
                "reason": "exception",
                "detail": {"traceback": traceback.format_exc(limit=10)},
            }
        )
        raise
