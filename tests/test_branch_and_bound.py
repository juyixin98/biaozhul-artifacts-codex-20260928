from __future__ import annotations

from pathlib import Path

from milp.model import parse_problem
from milp.response import build_response
from milp.schemas import LPProblem, SolveOptions
from milp.solver import solve
from tests.brute_force import oracle


def reasons(result):
    return [event["reason"] for event in result.events]


def test_root_infeasible_model_is_classified(settings):
    problem = LPProblem(
        name="root-infeasible",
        sense="minimize",
        objective={"x": 1},
        variables=[{"name": "x", "type": "integer", "lower": 0, "upper": 1}],
        constraints=[{"name": "impossible", "terms": {"x": 1}, "sense": ">=", "rhs": 2}],
    )
    result = solve(problem, SolveOptions(run_id="root-infeasible"), settings=settings)

    assert result.status == "infeasible"
    assert result.termination_reason == "lp_infeasible"
    assert result.objective_value is None
    assert result.best_bound is None
    assert reasons(result) == ["lp_infeasible"]


def test_relaxation_feasible_but_integer_infeasible(settings):
    problem = LPProblem(
        name="integer-empty",
        sense="minimize",
        objective={"x": 1},
        variables=[{"name": "x", "type": "integer", "lower": 0, "upper": 2}],
        constraints=[{"name": "fraction-only", "terms": {"x": 1}, "sense": "==", "rhs": 0.5}],
    )
    result = solve(problem, SolveOptions(run_id="integer-empty"), settings=settings)

    best, optimizers = oracle(problem)
    assert best is None and optimizers == []
    assert result.status == "infeasible"
    assert result.termination_reason == "lp_infeasible"
    assert result.objective_value is None
    assert "fractional_relaxation" in reasons(result)
    assert reasons(result).count("lp_infeasible") == 2
    assert set(event["event"] for event in result.events if event["reason"] == "fractional_integer_variable") == {"branch"}


def test_empty_integer_domain_is_infeasible_without_lp(settings):
    problem = LPProblem(
        name="empty-domain",
        sense="minimize",
        objective={"x": 1},
        variables=[{"name": "x", "type": "integer", "lower": 0.2, "upper": 0.8}],
    )
    result = solve(problem, SolveOptions(run_id="empty-domain"), settings=settings)

    assert result.status == "infeasible"
    assert result.termination_reason == "empty_integer_domain"
    assert result.nodes_solved == 0
    assert reasons(result) == ["empty_integer_domain"]


def test_unbounded_continuous_relaxation_is_not_optimal(settings):
    problem = LPProblem(
        name="continuous-unbounded",
        sense="minimize",
        objective={"y": -1},
        variables=[{"name": "y", "type": "continuous", "lower": 0}],
    )
    result = solve(problem, SolveOptions(run_id="continuous-unbounded"), settings=settings)

    assert result.status == "unbounded"
    assert result.termination_reason == "lp_unbounded"
    assert result.objective_value is None
    assert result.best_bound is None
    assert reasons(result) == ["lp_unbounded"]


def test_integer_tolerance_candidate_is_independently_rejected(settings):
    problem = LPProblem(
        name="near-integer-infeasible",
        sense="minimize",
        objective={"x": 1},
        variables=[{"name": "x", "type": "integer", "lower": 0, "upper": 2}],
        constraints=[{"name": "fraction-only", "terms": {"x": 1}, "sense": "==", "rhs": 1 - 5e-7}],
    )
    result = solve(
        problem,
        SolveOptions(run_id="near-integer", integer_tolerance=1e-6, constraint_tolerance=1e-8),
        settings=settings,
    )

    assert result.status == "infeasible"
    assert result.rejected_integer_candidate is True
    assert "equality_violation" in reasons(result)
    rejected = next(event for event in result.events if event["event"] == "candidate_rejected")
    assert rejected["detail"]["diagnostics"]["max_equality_residual"] > 1e-8


def test_post_lp_bound_pruning_after_incumbent(settings):
    problem = LPProblem(
        name="bound-prune",
        sense="minimize",
        objective={"x": -1, "y": -1},
        variables=[
            {"name": "x", "type": "binary"},
            {"name": "y", "type": "binary"},
        ],
        constraints=[{"name": "fractional-root", "terms": {"x": 1, "y": 1}, "sense": "<=", "rhs": 1.5}],
    )
    result = solve(problem, SolveOptions(run_id="bound-prune"), settings=settings)

    assert result.status == "optimal"
    assert result.objective_value == -1.0
    assert "bound" in reasons(result)
    bound_events = [event for event in result.events if event["reason"] == "bound"]
    assert any(event["detail"] == {} for event in bound_events)


def test_node_recovery_contains_constraints_and_bounds(settings):
    problem = LPProblem(
        name="recovery",
        sense="maximize",
        objective={"x": 5, "y": 3, "z": 2},
        variables=[
            {"name": "x", "type": "binary"},
            {"name": "y", "type": "binary"},
            {"name": "z", "type": "binary"},
        ],
        constraints=[{"name": "weight", "terms": {"x": 4, "y": 3, "z": 2}, "sense": "<=", "rhs": 5}],
    )
    result = solve(problem, SolveOptions(run_id="recovery", node_limit=1), settings=settings)
    response = build_response(result, parse_problem(problem).names, settings.max_events_in_response)

    assert result.status == "resource_exhausted"
    assert response.nodes_open
    node = response.open_nodes[0]
    assert node.depth == 1
    assert node.bounds
    assert node.branch_constraints
    assert {"variable", "operator", "value", "parent_node"} <= set(node.branch_constraints[0])


def test_journal_replays_run_with_problem_events_and_result(settings):
    problem = LPProblem(
        name="journal-problem",
        sense="minimize",
        objective={"x": 1},
        variables=[{"name": "x", "type": "integer", "lower": 0, "upper": 1}],
    )
    result = solve(problem, SolveOptions(run_id="journal-check"), settings=settings)
    lines = Path(result.journal_path).read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) >= 3
    assert '"record": "run_start"' in lines[0]
    assert '"name": "journal-problem"' in lines[0]
    assert any('"event": "lp_solved"' in line for line in lines)
    assert '"record": "run_complete"' in lines[-1]
