from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import solve as scipy_solve

from admm_consensus.config import Settings
from admm_consensus.coordinator import Coordinator
from admm_consensus.errors import ComputationError, InputError, StateConflict
from admm_consensus.kernel import initial_state_values, solve_local_update
from admm_consensus.models import Problem, QuadraticModel, SolverConfig, Tolerances, AdaptiveRho
from admm_consensus.parser import parse_problem
from admm_consensus.repository import RunRepository


def make_problem(two_worker_problem):
    Q1, Q2 = two_worker_problem["Q"]
    q1, q2 = two_worker_problem["q"]
    return Problem(
        models=(QuadraticModel(Q=Q1, q=q1), QuadraticModel(Q=Q2, q=q2)),
        dimension=3,
    )


def independent_reference(two_worker_problem):
    A1, A2 = two_worker_problem["A"]
    b1, b2 = two_worker_problem["b"]
    A = np.vstack([A1, A2])
    b = np.concatenate([b1, b2])
    Q_ref, q_ref = A.T @ A, -A.T @ b
    x_ref = scipy_solve(Q_ref, -q_ref, assume_a="pos")
    # The accepted model omits the data-fit constant 0.5*||b||^2.
    f_ref = 0.5 * x_ref @ Q_ref @ x_ref + q_ref @ x_ref
    return x_ref, f_ref


def make_coordinator(tmp_path):
    settings = Settings(data_dir=tmp_path, max_dimension=16, max_runs=10, max_iterations=100, max_workers=2)
    return Coordinator(RunRepository(settings))


def make_config(tmp_path, *, iterations=100, rho=0.7, adaptive=None):
    if rho == 0.7 and adaptive is None:
        rho = 5.0
        iterations = max(iterations, 60)
    return SolverConfig(
        max_iterations=iterations,
        initial_rho=rho,
        tol=Tolerances(primal_abs=1e-8, dual_abs=1e-8, primal_rel=1e-8, dual_rel=1e-8),
        adaptive_rho=adaptive,
    )


def test_asymmetric_problem_matches_independent_centralized_solution(tmp_path, two_worker_problem):
    coordinator = make_coordinator(tmp_path)
    problem = make_problem(two_worker_problem)
    x_ref, f_ref = independent_reference(two_worker_problem)
    state = coordinator.create_run(problem, make_config(tmp_path))

    while state.status == "running":
        state = coordinator.auto_round(state)

    assert state.status == "converged"
    np.testing.assert_allclose(state.z, x_ref, atol=5e-7)
    assert math.isclose(problem.total_objective(state.z), f_ref, rel_tol=1e-10, abs_tol=1e-9)
    final = state.trajectory[-1]
    assert final["primal_residual"] <= final["primal_threshold"]
    assert final["dual_residual"] <= final["dual_threshold"]
    # Both per-worker consensus disagreements must be small.
    assert np.max(np.linalg.norm(np.asarray(final["primal_vectors"]), axis=1)) < 5e-7


def test_fixed_update_order_and_interrupted_recovery(tmp_path, two_worker_problem):
    coordinator = make_coordinator(tmp_path)
    state = coordinator.create_run(make_problem(two_worker_problem), make_config(tmp_path))

    x1 = solve_local_update(state.problem.models[0], z=state.z, u=state.u[0], rho=state.rho, worker_id=1)
    coordinator.submit(state, 1, x1, round_number=1, rho=state.rho)

    # One submission alone never changes the round or consensus variables.
    reloaded = coordinator.get(state.run_id)
    assert reloaded.round_number == 1
    np.testing.assert_allclose(reloaded.z, np.zeros(3))
    assert set(reloaded.pending) == {1}

    # A stale submission from another round is rejected before mutation.
    with pytest.raises(StateConflict) as conflict:
        x2 = solve_local_update(state.problem.models[1], z=state.z, u=state.u[1], rho=state.rho, worker_id=2)
        coordinator.submit(reloaded, 2, x2, round_number=2, rho=state.rho)
    assert conflict.value.details["submitted_round"] == 2

    coordinator.submit(reloaded, 2, x2, round_number=1, rho=state.rho)
    coordinator.complete_round(reloaded)
    recovered = coordinator.get(state.run_id)
    assert recovered.round_number == 2
    np.testing.assert_allclose(recovered.z, np.mean(np.vstack([x1, x2]), axis=0))
    expected_u1 = x1 - recovered.z
    expected_u2 = x2 - recovered.z
    np.testing.assert_allclose(recovered.u[0], expected_u1)
    np.testing.assert_allclose(recovered.u[1], expected_u2)


def test_rho_change_rescales_dual_before_next_local_round(tmp_path, two_worker_problem):
    adaptive = AdaptiveRho(mu=1.5, tau_incr=2.5, tau_decr=2.0, change_rounds=frozenset({1}))
    coordinator = make_coordinator(tmp_path)
    state = coordinator.create_run(make_problem(two_worker_problem), make_config(tmp_path, rho=0.2, adaptive=adaptive))

    state = coordinator.auto_round(state)
    assert state.round_number == 2
    assert math.isclose(state.rho, 0.5, rel_tol=1e-12)
    expected_u = [0.2 / 0.5 * np.asarray(value) for value in state.trajectory[-1]["u"]]
    np.testing.assert_allclose(state.u[0], expected_u[0])
    np.testing.assert_allclose(state.u[1], expected_u[1])

    # Persistence and worker task carry the same next-round rho.
    reloaded = coordinator.get(state.run_id)
    assert math.isclose(reloaded.rho, state.rho)
    task = coordinator.snapshot(reloaded)
    assert math.isclose(task["next_round_rho"], 0.5)


def test_timeout_rejects_entire_round_and_records_reason(tmp_path, two_worker_problem):
    coordinator = make_coordinator(tmp_path)
    state = coordinator.create_run(make_problem(two_worker_problem), make_config(tmp_path))
    x1 = solve_local_update(state.problem.models[0], z=state.z, u=state.u[0], rho=state.rho, worker_id=1)
    coordinator.submit(state, 1, x1, round_number=1, rho=state.rho)
    coordinator.timeout_round(state, missing_worker_ids=[2], reason="unit-test-deadline")

    assert state.round_number == 1
    assert state.pending == {}
    assert state.trajectory == []
    assert state.last_timeout["missing_worker_ids"] == [2]
    assert state.last_timeout["discarded_worker_ids"] == [1]

    recovered = coordinator.get(state.run_id)
    assert recovered.round_number == 1
    assert recovered.last_timeout["reason"] == "unit-test-deadline"
    events = coordinator.repository.events(state.run_id)
    timeout_events = [event for event in events if event["event_type"] == "round_timeout"]
    assert len(timeout_events) == 1
    assert timeout_events[0]["details"]["decision"].startswith("whole_round_rejected")


def test_max_iterations_reports_unconverged_state(tmp_path, two_worker_problem):
    coordinator = make_coordinator(tmp_path)
    state = coordinator.create_run(
        make_problem(two_worker_problem),
        make_config(tmp_path, iterations=2, rho=0.01),
    )
    state = coordinator.auto_round(state)
    assert state.round_number == 2
    state = coordinator.auto_round(state)
    assert state.status == "max_iterations"
    final = state.trajectory[-1]
    assert final["termination"].startswith("not converged")
    assert len(state.trajectory) == 2


def test_input_and_computation_failure_categories_are_distinct(tmp_path):
    settings = Settings(data_dir=tmp_path, max_dimension=8, max_runs=10, max_iterations=100, max_workers=2)
    with pytest.raises(InputError) as bad_shape:
        parse_problem(
            {
                "dimension": 2,
                "workers": [
                    {"worker_id": 1, "Q": np.eye(2).tolist(), "q": [0.0, 0.0]},
                    {"worker_id": 2, "Q": np.eye(2).tolist(), "q": [0.0]},
                ],
            },
            settings,
        )
    assert bad_shape.value.code == "input"

    model = QuadraticModel(Q=np.eye(2), q=np.zeros(2))
    with pytest.raises(ComputationError) as bad_rho:
        solve_local_update(model, z=np.zeros(2), u=np.zeros(2), rho=float("nan"), worker_id=1)
    assert bad_rho.value.code == "computation"
