import numpy as np
import pytest
from scipy.optimize import linprog

from entropic_ot.errors import ComputationFailedError, ResourceExhaustedError
from entropic_ot.solver import solve_entropic_ot
from entropic_ot.validation import validate_problem


def solve(a, b, c, epsilon, *, max_iterations=10_000, atol=1e-10, rtol=1e-8):
    problem = validate_problem(
        a,
        b,
        c,
        epsilon=epsilon,
        max_iterations=max_iterations,
    )
    return solve_entropic_ot(problem, run_id="unit-run", atol=atol, rtol=rtol)


def assert_marginals(plan, a, b, *, rtol=1e-8, atol=1e-10):
    np.testing.assert_allclose(plan.sum(axis=1), a, rtol=rtol, atol=atol)
    np.testing.assert_allclose(plan.sum(axis=0), b, rtol=rtol, atol=atol)


def test_two_by_two_closed_form_plan_and_separate_objectives():
    # Marginals: x11=x22=0.25-t, x12=t, x21=0.5+t.
    # Interior stationarity gives t^2/((0.25-t)(0.5+t))=exp(-8).
    q = np.exp(-8.0)
    t = (np.sqrt(1.0 + 3.0 * q) - (1.0 + q)) / (4.0 * (1.0 - q))
    expected = np.array(
        [
            [0.25 - t, t],
            [0.5 + t, 0.25 - t],
        ]
    )
    cost = np.array([[0.0, 2.0], [2.0, 0.0]])

    result = solve([0.25, 0.75], [0.75, 0.25], cost, 0.5, atol=1e-12, rtol=1e-10)

    np.testing.assert_allclose(result.plan, expected, rtol=1e-9, atol=1e-12)
    assert_marginals(result.plan, [0.25, 0.75], [0.75, 0.25], rtol=1e-10, atol=1e-12)
    expected_cost = float(np.sum(expected * cost))
    expected_regularized = expected_cost + 0.5 * np.sum(
        expected * (np.log(expected) - 1.0)
    )
    assert result.transport_cost == pytest.approx(expected_cost, rel=1e-9)
    assert result.regularized_objective == pytest.approx(expected_regularized, abs=1e-9)
    assert result.regularized_objective != pytest.approx(result.transport_cost)
    assert result.diagnostics.feasible is True


def test_large_cost_and_tiny_regularization_does_not_underflow_or_overflow():
    cost = np.array([[0.0, 1e12], [1e12, 0.0]])

    result = solve([0.5, 0.5], [0.5, 0.5], cost, 1e-8)

    assert np.all(np.isfinite(result.plan))
    np.testing.assert_allclose(result.plan, np.diag([0.5, 0.5]), atol=1e-14)
    assert_marginals(result.plan, [0.5, 0.5], [0.5, 0.5])
    assert result.transport_cost == pytest.approx(0.0, abs=1e-14)
    positive_plan = result.plan[result.plan > 0]
    expected_entropy = float(np.sum(positive_plan * (np.log(positive_plan) - 1.0)))
    assert result.regularized_objective == pytest.approx(
        1e-8 * expected_entropy,
        abs=1e-12,
    )


def test_extreme_off_diagonal_edges_materialize_as_exact_zeros_without_nan():
    cost = np.array([[0.0, 7000.0], [7000.0, 0.0]])

    result = solve([0.5, 0.5], [0.5, 0.5], cost, 1.0)

    assert np.all(np.isfinite(result.plan))
    np.testing.assert_array_equal(result.plan, np.diag([0.5, 0.5]))
    assert result.diagnostics.events[-1].finite_log_plan is True


def test_zero_mass_rows_and_columns_are_exactly_zero_and_active_plan_preserved():
    a = [0.0, 0.0, 1.0, 0.0]
    b = [0.4, 0.6, 0.0, 0.0]
    cost = np.array(
        [
            [99.0, 99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0, 99.0],
            [1.0, 5.0, 99.0, 99.0],
            [99.0, 99.0, 99.0, 99.0],
        ]
    )

    result = solve(a, b, cost, 0.05)

    assert result.plan.shape == (4, 4)
    np.testing.assert_array_equal(result.plan[0:2, :], 0.0)
    np.testing.assert_array_equal(result.plan[3, :], 0.0)
    np.testing.assert_array_equal(result.plan[:, 2:4], 0.0)
    np.testing.assert_allclose(result.plan[2, 0], 0.4, atol=1e-12)
    np.testing.assert_allclose(result.plan[2, 1], 0.6, atol=1e-12)
    assert result.transport_cost == pytest.approx(3.4, abs=1e-10)
    expected_kl = 0.4 * np.log(0.4 * 16) + 0.6 * np.log(0.6 * 16)
    assert result.kl_to_uniform == pytest.approx(expected_kl, rel=1e-10)


def test_all_zero_masses_return_zero_feasible_plan_without_iteration():
    result = solve([0.0, 0.0], [0.0, 0.0], [[1.0, 2.0], [3.0, 4.0]], 0.2)

    np.testing.assert_array_equal(result.plan, np.zeros((2, 2)))
    assert result.transport_cost == 0.0
    assert result.regularized_objective == 0.0
    assert result.diagnostics.feasible is True
    assert result.diagnostics.iterations == 0
    assert result.diagnostics.stopped_reason == "all_masses_zero"


def test_non_unit_total_mass_conserves_scale_and_objective():
    cost = np.array([[1.0, 2.0], [2.0, 1.0]])
    result = solve([2.0, 3.0], [1.5, 3.5], cost, 0.25)

    assert_marginals(result.plan, [2.0, 3.0], [1.5, 3.5])
    assert result.plan.sum() == pytest.approx(5.0, rel=1e-10)
    positive = result.plan[result.plan > 0]
    expected_entropy = float(np.sum(positive * (np.log(positive) - 1.0)))
    expected_cost = float(np.sum(result.plan * cost))
    assert result.transport_cost == pytest.approx(expected_cost, rel=1e-12)
    assert result.regularized_objective == pytest.approx(
        expected_cost + 0.25 * expected_entropy,
        rel=1e-12,
    )


def test_small_problem_matches_independent_lp_reference():
    a = np.array([0.2, 0.3, 0.5])
    b = np.array([0.1, 0.4, 0.5])
    cost = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 0.0, 1.0],
            [1.0, 2.0, 0.0],
        ]
    )

    row_equalities = np.kron(np.eye(3), np.ones(3)).reshape(3, 9)
    column_equalities = np.kron(np.ones(3), np.eye(3))
    lp = linprog(
        cost.ravel(),
        A_eq=np.vstack([row_equalities, column_equalities]),
        b_eq=np.r_[a, b],
        bounds=[(0.0, None)] * 9,
        method="highs",
    )
    assert lp.success

    result = solve(a, b, cost, 0.01, max_iterations=20_000)
    expected_plan = lp.x.reshape(3, 3)

    assert_marginals(result.plan, a, b)
    assert result.transport_cost == pytest.approx(lp.fun, abs=1e-7)
    assert result.regularized_objective < result.transport_cost
    # Sinkhorn can spread mass where LP places exact zeros, but costs still agree.
    np.testing.assert_allclose(result.plan.sum(), expected_plan.sum(), rtol=1e-10)


def test_iteration_budget_failure_is_not_marked_feasible():
    cost = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 0.0, 1.0],
            [1.0, 2.0, 0.0],
        ]
    )

    with pytest.raises(ResourceExhaustedError) as exc_info:
        solve(
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
            cost,
            0.01,
            max_iterations=3,
        )

    error = exc_info.value
    assert error.category == "resource_exhausted"
    assert error.status_code == 503
    assert error.diagnostics["feasible"] is False
    assert error.diagnostics["converged"] is False
    assert error.diagnostics["stopped_reason"] == "iteration_budget_exhausted"
    assert error.diagnostics["iterations"] == 3
    assert error.diagnostics["events"][-1]["iteration"] == 3


def test_non_finite_numerics_are_reported_as_computation_failure(monkeypatch):
    import entropic_ot.solver as solver_module

    def broken_logsumexp(values, axis):
        return np.full(values.shape[:-1], np.inf)

    monkeypatch.setattr(solver_module, "logsumexp", broken_logsumexp)

    with pytest.raises(ComputationFailedError) as exc_info:
        solve(
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
            np.zeros((3, 3)),
            0.1,
            max_iterations=10,
        )

    assert exc_info.value.category == "computation_failed"
    assert exc_info.value.status_code == 500
    assert "non-finite" in exc_info.value.message
    assert exc_info.value.diagnostics["run_id"] == "unit-run"
    assert exc_info.value.diagnostics["feasible"] is False
    assert exc_info.value.diagnostics["active_shape"] == [3, 3]
