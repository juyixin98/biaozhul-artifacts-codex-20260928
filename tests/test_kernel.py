import math

import numpy as np

from lp_service.config import SolverConfig
from lp_service.parser import parse_request
from lp_service.solver import solve_lp
from lp_service.standardize import standardize
from lp_service.kernel import two_phase_simplex

from .fixtures import (
    bounded_and_fixed,
    degenerate_beale_cycling,
    infeasible_with_certificate,
    redundant_constraints,
    two_variable_optimal,
    unbounded_with_free_upper,
)


def test_hand_computed_optimal_point_and_dual_evidence():
    result = solve_lp(two_variable_optimal(), request_id="hand-optimal").response
    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(2.0)
    np.testing.assert_allclose(result.solution, [0.0, 2.0], atol=1e-8)
    assert result.duals["constraints"][0] == pytest.approx(0.5)
    assert result.diagnostic.accepted is True
    assert result.diagnostic.checks["weak_duality_gap_zero"] is True


def test_bland_rule_terminates_on_hand_computed_degenerate_case():
    result = solve_lp(degenerate_beale_cycling(), request_id="beale").response
    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(-0.395)
    np.testing.assert_allclose(result.solution, [0.5, 0.0, 1.0, 0.0], atol=1e-8)
    assert result.diagnostic.accepted is True


def test_redundant_equalities_leave_artificial_variables_and_remain_solvable():
    problem = parse_request(redundant_constraints())
    model = standardize(problem)
    kernel = two_phase_simplex(model, SolverConfig(), request_id="redundant")
    assert kernel.status == "optimal"
    assert sum(event.get("event") == "redundant_rows_removed" for event in kernel.history) == 1
    result = solve_lp(redundant_constraints(), request_id="redundant").response
    np.testing.assert_allclose(result.solution, [1.0, 2.0], atol=1e-8)
    assert result.objective_value == pytest.approx(-1.0)
    assert result.diagnostic.accepted is True


def test_infeasible_returns_specific_farkas_failure_category_not_just_code():
    result = solve_lp(infeasible_with_certificate(), request_id="infeasible").response
    assert result.status == "infeasible"
    assert result.solution is None
    q = result.certificate["constraints"]
    assert q[0] == pytest.approx(1.0)
    assert q[1] == pytest.approx(1.0)
    assert result.certificate["contradiction"] == pytest.approx(1.0, abs=1e-8)
    assert result.diagnostic.checks["farkas_contradiction_positive"] is True
    assert result.diagnostic.accepted is True


def test_unbounded_returns_verified_ray():
    result = solve_lp(unbounded_with_free_upper(), request_id="unbounded").response
    assert result.status == "unbounded"
    d = np.asarray(result.unbounded_direction)
    np.testing.assert_allclose(np.asarray([2.0, -1.0]) @ d, 0.0, atol=1e-8)
    assert np.all(d >= -1e-8)
    assert np.dot([-1.0, -1.0], d) < -1e-6
    assert result.diagnostic.accepted is True


def test_fixed_variable_and_finite_bounds_are_mapped_back():
    result = solve_lp(bounded_and_fixed(), request_id="fixed-bounds").response
    assert result.status == "optimal"
    np.testing.assert_allclose(result.solution, [1.0, 2.0], atol=1e-8)
    assert result.objective_value == pytest.approx(-1.0)
    assert result.diagnostic.accepted is True


def test_numerical_uncertain_is_distinct_from_optimal_and_infeasible():
    config = SolverConfig(max_iterations=1)
    problem = parse_request(degenerate_beale_cycling())
    model = standardize(problem)
    kernel = two_phase_simplex(model, config, request_id="cap")
    assert kernel.status == "numerical_uncertain"
    assert math.isfinite(kernel.history[-1]["reason"] is not None)


import pytest
