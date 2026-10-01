from __future__ import annotations

import math

import numpy as np
import pytest

from entropic_ot.diagnostics import RunRecorder
from entropic_ot.errors import (
    ComputationFailedError,
    InputQualityError,
    ResourceExhaustedError,
)
from entropic_ot.kernel import solve_entropic_ot

from .references import linear_program_ot, multiplicative_sinkhorn


def test_two_point_distribution_matches_hand_computed_solution() -> None:
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])

    result = solve_entropic_ot(a, b, cost, epsilon=1.0, tolerance=1e-11)

    expected_offdiagonal = 0.5 / (1.0 + math.e)
    expected_plan = np.array(
        [
            [0.5 - expected_offdiagonal, expected_offdiagonal],
            [expected_offdiagonal, 0.5 - expected_offdiagonal],
        ]
    )
    expected_entropy_penalty = (
        2.0 * expected_plan[0, 0] * math.log(4.0 * expected_plan[0, 0])
        + 2.0 * expected_plan[0, 1] * math.log(4.0 * expected_plan[0, 1])
    )

    assert result.converged is True
    assert result.feasible is True
    assert result.stop_reason == "converged"
    assert result.iterations == 1
    np.testing.assert_allclose(result.plan, expected_plan, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(result.original_cost, 2.0 * expected_offdiagonal)
    np.testing.assert_allclose(
        result.regularized_cost,
        2.0 * expected_offdiagonal + expected_entropy_penalty,
        rtol=1e-12,
    )
    np.testing.assert_allclose(result.row_marginals, a)
    np.testing.assert_allclose(result.column_marginals, b)


def test_small_epsilon_agrees_with_independent_lp() -> None:
    rng = np.random.default_rng(7)
    a = rng.dirichlet(np.ones(4))
    b = rng.dirichlet(np.ones(4))
    cost = rng.random((4, 4))

    result = solve_entropic_ot(
        a, b, cost, epsilon=1e-5, max_iterations=100_000, tolerance=1e-10
    )
    lp_plan, lp_cost = linear_program_ot(a, b, cost)

    assert result.converged is True
    assert result.feasible is True
    assert result.original_cost == pytest.approx(lp_cost, abs=5e-6)
    np.testing.assert_allclose(result.plan.sum(axis=1), a, atol=1e-9)
    np.testing.assert_allclose(result.plan.sum(axis=0), b, atol=1e-9)
    assert np.max(np.abs(result.plan - lp_plan)) < 0.02


def test_matches_classical_multiplicative_reference() -> None:
    a = np.array([0.2, 0.3, 0.5])
    b = np.array([0.25, 0.35, 0.4])
    cost = np.array(
        [
            [0.0, 0.4, 0.9],
            [0.3, 0.1, 0.6],
            [0.8, 0.5, 0.2],
        ]
    )

    result = solve_entropic_ot(a, b, cost, epsilon=0.08, tolerance=1e-10)
    reference_plan, _ = multiplicative_sinkhorn(
        a, b, cost, epsilon=0.08, tolerance=1e-9
    )

    assert result.converged is True
    np.testing.assert_allclose(result.plan, reference_plan, atol=1e-8)


def test_zero_rows_and_columns_are_retained_and_excluded() -> None:
    a = np.array([0.25, 0.0, 0.25, 0.5])
    b = np.array([0.4, 0.0, 0.6])
    cost = np.array(
        [
            [0.1, 99.0, 0.7],
            [7.0, 8.0, 9.0],
            [0.5, 1.0, 0.2],
            [0.3, 2.0, 0.4],
        ]
    )

    result = solve_entropic_ot(a, b, cost, epsilon=0.02, tolerance=1e-10)

    assert result.converged is True
    assert result.feasible is True
    assert result.diagnostics["active_rows"] == [0, 2, 3]
    assert result.diagnostics["active_columns"] == [0, 2]
    assert result.plan.shape == (4, 3)
    assert np.all(result.plan[1, :] == 0.0)
    assert np.all(result.plan[:, 1] == 0.0)
    np.testing.assert_allclose(result.plan.sum(axis=1), a, atol=1e-10)
    np.testing.assert_allclose(result.plan.sum(axis=0), b, atol=1e-10)


def test_zero_total_mass_returns_zero_plan_without_calling_sinkhorn() -> None:
    result = solve_entropic_ot(
        np.zeros(3),
        np.zeros(3),
        np.zeros((3, 3)),
        epsilon=0.1,
    )

    assert result.converged is True
    assert result.feasible is True
    assert result.stop_reason == "zero_total_mass"
    assert result.iterations == 0
    assert np.count_nonzero(result.plan) == 0
    assert result.original_cost == 0.0
    assert result.regularized_cost == 0.0


@pytest.mark.parametrize(
    ("a", "b", "cost", "message"),
    [
        ([-0.1, 1.1], [0.5, 0.5], [[0, 1], [1, 0]], "non-negative"),
        ([0.5, 0.5], [0.4, 0.5], [[0, 1], [1, 0]], "masses"),
        ([0.5, 0.5], [0.5, 0.5, 0.0], [[0, 1], [1, 0]], "shape"),
        ([0.5, 0.5], [0.5, 0.5], [[0, 1]], "shape"),
        ([0.5, np.nan], [0.5, 0.5], [[0, 1], [1, 0]], "finite"),
    ],
)
def test_input_errors_are_classified(a, b, cost, message) -> None:
    with pytest.raises(InputQualityError, match=message):
        solve_entropic_ot(a, b, cost, epsilon=0.1)


def test_large_cost_small_epsilon_avoids_global_underflow() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    cost = np.array([[1000.0, 0.0], [0.0, 1000.0]])

    result = solve_entropic_ot(
        a, b, cost, epsilon=1e-6, max_iterations=10, tolerance=1e-10
    )

    assert result.converged is True
    assert result.feasible is True
    np.testing.assert_allclose(result.plan, [[0.0, 1.0], [0.0, 0.0]], atol=1e-14)
    assert result.original_cost == pytest.approx(0.0, abs=1e-14)
    assert np.all(np.isfinite(result.plan))


def test_expired_budget_is_not_reported_as_feasible() -> None:
    a = np.array([0.7, 0.3])
    b = np.array([0.4, 0.6])
    cost = np.array([[0.0, 100.0], [100.0, 0.0]])

    result = solve_entropic_ot(
        a, b, cost, epsilon=1e-8, max_iterations=1, tolerance=1e-10
    )

    assert result.converged is False
    assert result.feasible is False
    assert result.stop_reason == "iteration_budget_exhausted"
    np.testing.assert_allclose(result.row_marginals, a)
    assert result.column_error > 0.1
    assert result.diagnostics["feasibility_reason"] != (
        "both reconstructed row and column marginals satisfy their tolerances"
    )


def test_computation_failure_is_distinct_from_resource_exhaustion(monkeypatch) -> None:
    def broken_logsumexp(*args, **kwargs):
        if kwargs.get("axis") == 0:
            raise ArithmeticError("forced numerical failure")
        return np.array([0.0, 0.0])

    monkeypatch.setattr("entropic_ot.kernel.logsumexp", broken_logsumexp)
    with pytest.raises(ComputationFailedError) as exc_info:
        solve_entropic_ot(
            [0.5, 0.5],
            [0.5, 0.5],
            [[0.0, 1.0], [1.0, 0.0]],
            epsilon=1.0,
        )
    assert exc_info.value.category == "computation_failed"


def test_memory_error_is_reported_as_resource_exhaustion(monkeypatch) -> None:
    def broken_exp(*args, **kwargs):
        raise MemoryError("forced allocation failure")

    monkeypatch.setattr("entropic_ot.kernel.np.exp", broken_exp)
    with pytest.raises(ResourceExhaustedError) as exc_info:
        solve_entropic_ot(
            [0.5, 0.5],
            [0.5, 0.5],
            [[0.0, 1.0], [1.0, 0.0]],
            epsilon=1.0,
        )
    assert exc_info.value.category == "resource_exhausted"


def test_run_log_contains_replay_id_intermediates_and_final_judgment(tmp_path) -> None:
    recorder = RunRecorder(tmp_path, run_id="replay-001")
    result = solve_entropic_ot(
        [0.5, 0.5],
        [0.5, 0.5],
        [[0.0, 1.0], [1.0, 0.0]],
        epsilon=1.0,
        max_iterations=2,
        tolerance=1e-10,
        recorder=recorder,
        include_logged_inputs=True,
    )

    events = recorder.read_events()
    assert [event["event"] for event in events] == ["start", "iteration", "finish"]
    assert events[0]["run_id"] == "replay-001"
    assert events[0]["inputs"]["a"] == [0.5, 0.5]
    assert "log_column_error" in events[1]["state"]
    assert events[2]["result"]["feasible"] is result.feasible
    assert "feasibility_reason" in events[2]["result"]["diagnostics"]
