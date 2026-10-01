from __future__ import annotations

from milp.model import parse_problem
from milp.schemas import SolveOptions
from milp.solver import solve
from tests.brute_force import knapsack, oracle


def test_knapsack_matches_exhaustive_oracle(settings):
    problem = knapsack()
    result = solve(problem, SolveOptions(run_id="oracle-knapsack"), settings=settings)

    best, optimizers = oracle(problem)
    assert result.status == "optimal"
    assert result.termination_reason == "all_open_nodes_pruned"
    assert result.objective_value == best
    assert result.incumbent is not None
    solution = {name: float(value) for name, value in zip(problem.objective, result.incumbent, strict=False)}
    assert solution in optimizers or abs(result.objective_value or 0.0) - abs(best or 0.0) == 0.0
    assert result.absolute_gap == 0.0
    assert result.best_bound == best


def test_node_limit_returns_incumbent_and_true_gap(settings):
    problem = knapsack()
    result = solve(problem, SolveOptions(run_id="node-limit-gap", node_limit=6), settings=settings)

    best, _ = oracle(problem)
    assert result.status == "resource_exhausted"
    assert result.termination_reason == "node_limit"
    assert result.objective_value == 5.0
    assert result.best_bound is not None
    assert result.objective_value is not None
    assert result.best_bound == 6.0
    assert result.absolute_gap == 1.0
    assert best == 5.0


def test_time_limit_zero_is_resource_exhausted_not_optimal(settings):
    problem = knapsack()
    result = solve(
        problem,
        SolveOptions(run_id="time-limit-zero", time_limit_seconds=0.0),
        settings=settings,
    )

    assert result.status == "resource_exhausted"
    assert result.termination_reason == "time_limit"
    assert result.objective_value is None
    assert result.absolute_gap is None
    assert len(result.open_nodes) >= 1
