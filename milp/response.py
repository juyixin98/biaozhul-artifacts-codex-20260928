"""Conversion of internal solver results into HTTP response models."""

from __future__ import annotations

from typing import Any

from .schemas import Event, NodeState, SolveResponse
from .solver import SolveResult


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _finite_or_none(value: float) -> float | None:
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return 0.0 if value == 0.0 else float(value)


def build_response(result: SolveResult, problem_names: list[str], max_events: int) -> SolveResponse:
    objective = result.objective_value
    best_bound = result.best_bound
    absolute_gap = result.absolute_gap
    relative_gap: float | None = None
    if objective is not None and best_bound is not None:
        denominator = max(1.0, abs(objective))
        relative_gap = absolute_gap / denominator

    solution = None
    if result.incumbent is not None:
        solution = {
            name: (0.0 if result.incumbent[i] == 0.0 else float(result.incumbent[i]))
            for i, name in enumerate(problem_names)
        }

    events = [Event(**_clean(event)) for event in result.events]
    events_omitted = max(0, len(events) - max_events)
    events = events[-max_events:]

    return SolveResponse(
        run_id=result.run_id,
        status=result.status,
        termination_reason=result.termination_reason,
        sense=result.sense,
        objective_value=objective,
        best_bound=best_bound,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        solution=solution,
        integer_infeasible_candidate_rejected=result.rejected_integer_candidate,
        nodes_solved=result.nodes_solved,
        nodes_open=len(result.open_nodes),
        events_omitted=events_omitted,
        events=events,
        open_nodes=[
            NodeState(
                node_id=node.node_id,
                parent_id=node.parent_id,
                depth=node.depth,
                branch_constraints=node.branch_constraints,
                bounds={
                    problem_names[index]: {
                        "lower": _finite_or_none(lower),
                        "upper": _finite_or_none(upper),
                    }
                    for index, (lower, upper) in sorted(node.bounds.items())
                },
                relaxation_status=node.relaxation_status,
                relaxation_objective=node.relaxation_objective,
                bound=_finite_or_none(node.bound),
            )
            for node in result.open_nodes
        ],
        diagnostics=_clean(result.diagnostics),
        journal_path=result.journal_path,
    )
