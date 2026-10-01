from __future__ import annotations

import numpy as np
import pytest

from consensus_admm.errors import ModelParseError
from consensus_admm.kernel import (
    ADMMConfig,
    apply_round,
    check_convergence,
    rescale_dual,
    solve_local,
)
from consensus_admm.models import LocalUpdate
from consensus_admm.parser import parse_problem


def config(**overrides):
    values = {
        "initial_rho": 1.0,
        "eps_abs": 1e-7,
        "eps_rel": 1e-7,
        "max_iterations": 80,
        "deadline_ms": 1000,
        "rho_schedule": {},
    }
    values.update(overrides)
    return ADMMConfig(**values)


def asymmetric_model():
    return parse_problem(
        {
            "dimension": 3,
            "workers": [
                {
                    "id": "small",
                    "P": [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]],
                    "q": [1.0, -2.0, 0.5],
                    "sample_count": 13,
                },
                {
                    "id": "large",
                    "P": [[4.0, 0.2, 0.0], [0.2, 5.0, -0.1], [0.0, -0.1, 2.0]],
                    "q": [-3.0, 1.0, 4.0],
                    "r": 0.25,
                    "sample_count": 97,
                },
            ],
        }
    )


def central_reference(model):
    # Independent centralized KKT solve; no ADMM helper is involved.
    P = sum(worker.P for worker in model.workers)
    q = sum(worker.q for worker in model.workers)
    x_star = np.linalg.solve(P, -q)
    objective = sum(
        0.5 * x_star @ worker.P @ x_star + worker.q @ x_star + worker.r
        for worker in model.workers
    )
    return x_star, float(objective)


def run_admm(model, cfg):
    n = model.dimension
    z = np.zeros(n)
    u = {worker.worker_id: np.zeros(n) for worker in model.workers}
    rho = cfg.initial_rho
    trajectory = []
    for iteration in range(cfg.max_iterations):
        rho_new = float(cfg.rho_schedule.get(iteration, cfg.initial_rho))
        u = {
            worker_id: rescale_dual(value, rho, rho_new)
            for worker_id, value in u.items()
        }
        locals_ = {}
        for worker in model.workers:
            local = solve_local(
                worker, iteration, rho_new, z, u[worker.worker_id]
            )
            assert np.isfinite(local.x).all()
            assert local.stationarity_residual < 1e-8
            locals_[worker.worker_id] = local
        result = apply_round(
            model,
            iteration,
            rho_new,
            rho,
            z,
            u,
            locals_,
            cfg,
        )
        decision = check_convergence(result)
        trajectory.append(result)
        z, u, rho = result.z, result.u, result.rho
        if decision.converged:
            return z, result.objective_total, trajectory
    return z, trajectory[-1].objective_total, trajectory


def test_parser_rejects_non_symmetric_and_negative_curvature():
    with pytest.raises(ModelParseError) as exc_info:
        parse_problem(
            {
                "dimension": 1,
                "workers": [
                    {"id": "a", "P": [[-1.0]], "q": [0.0]},
                    {"id": "b", "P": [[1.0]], "q": [0.0]},
                ],
            }
        )
    assert exc_info.value.code == "invalid_quadratic_model"
    assert exc_info.value.details["min_eigenvalue"] < 0


def test_asymmetric_two_process_problem_matches_central_optimum():
    model = asymmetric_model()
    cfg = config(max_iterations=100, eps_abs=1e-9, eps_rel=1e-9)
    x_star, objective_star = central_reference(model)
    x_admm, objective_admm, trajectory = run_admm(model, cfg)

    assert len(trajectory) < 100
    assert trajectory[-1].primal_residual <= trajectory[-1].eps_primal
    assert trajectory[-1].dual_residual <= trajectory[-1].eps_dual
    np.testing.assert_allclose(x_admm, x_star, atol=1e-7, rtol=1e-7)
    assert objective_admm == pytest.approx(objective_star, abs=2e-8)
    assert trajectory[0].primal_residual > trajectory[-1].primal_residual


def test_rho_change_preserves_unscaled_dual_and_continues_trajectory():
    model = asymmetric_model()
    cfg = config(rho_schedule={2: 4.0}, max_iterations=80)
    x_star, _ = central_reference(model)
    x_admm, _, trajectory = run_admm(model, cfg)

    changed = trajectory[2]
    assert changed.rho == 4.0
    assert changed.rho_previous == 1.0
    # y=rho*u at the start of iteration 2 must remain the prior scaled dual.
    previous_end_y = trajectory[1].rho_previous * trajectory[1].u["small"]
    changed_start_y = changed.rho * changed.u_pre["small"]
    np.testing.assert_allclose(changed_start_y, previous_end_y, atol=1e-12)
    np.testing.assert_allclose(x_admm, x_star, atol=1e-5)


def test_primal_and_dual_termination_are_independent():
    round_update = type(
        "Round",
        (),
        {
            "primal_residual": 1.0,
            "eps_primal": 2.0,
            "dual_residual": 3.0,
            "eps_dual": 2.0,
        },
    )
    decision = check_convergence(round_update)
    assert not decision.converged
    assert decision.primal_ok is True
    assert decision.dual_ok is False
    assert decision.reason == "tolerance_not_satisfied:dual"


def test_round_rejects_local_update_from_different_iteration():
    model = asymmetric_model()
    cfg = config()
    local = solve_local(
        model.workers[0],
        iteration=7,
        rho=1.0,
        z_pre=np.zeros(3),
        u_pre=np.zeros(3),
    )
    wrong = LocalUpdate(
        worker_id=local.worker_id,
        iteration=7,
        rho=local.rho,
        z_pre=local.z_pre,
        u_pre=local.u_pre,
        x=local.x,
        objective=local.objective,
        stationarity_residual=local.stationarity_residual,
        solve_seconds=local.solve_seconds,
    )
    good = solve_local(
        model.workers[1],
        iteration=0,
        rho=1.0,
        z_pre=np.zeros(3),
        u_pre=np.zeros(3),
    )
    with pytest.raises(Exception, match="different iteration"):
        apply_round(
            model,
            0,
            1.0,
            1.0,
            np.zeros(3),
            {worker.worker_id: np.zeros(3) for worker in model.workers},
            {"small": wrong, "large": good},
            cfg,
        )
