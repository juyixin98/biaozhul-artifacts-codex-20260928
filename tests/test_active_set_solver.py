import numpy as np

from conftest import (
    boundary_request,
    near_degenerate_request,
    negative_multiplier_request,
    rank_deficient_equalities_request,
    redundant_constraints_request,
    scipy_slsqp_reference,
)


def event_types(response):
    return [event["type"] for event in response["events"]]


def test_boundary_optimum_matches_independent_slsqp(solve):
    request = boundary_request()
    response = solve(request)
    reference = scipy_slsqp_reference(request)

    assert response["status"] == "optimal"
    assert response["failure_reasons"] == []
    np.testing.assert_allclose(response["solution"]["x"], [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(response["solution"]["x"], reference.x, atol=5e-8)
    assert response["kkt_residuals"]["objective"] == pytest_approx(-3.0)
    assert response["kkt_residuals"]["stationarity_inf"] < 1e-8
    assert response["kkt_residuals"]["primal_inequality_violation"] < 1e-8
    assert response["kkt_residuals"]["dual_feasibility_violation"] < 1e-8
    assert response["kkt_residuals"]["complementarity_max"] < 1e-8
    assert response["active_set"]["solver_active_inequality_indices"] == [0, 1]
    assert "negative_multiplier_constraint_removed" in event_types(response)
    assert "blocking_constraint_added" in event_types(response)


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=1e-10)


def test_redundant_constraints_do_not_enter_rank_deficient_face(solve):
    request = redundant_constraints_request()
    response = solve(request)
    reference = scipy_slsqp_reference(request)

    assert response["status"] == "optimal"
    np.testing.assert_allclose(response["solution"]["x"], reference.x, atol=1e-7)
    active_names = response["active_set"]["solver_active_inequality_names"]
    assert "binding_sum" in active_names
    assert "duplicate_sum_scaled" not in active_names
    assert "duplicate_sum_ge_form" not in active_names
    assert response["kkt_residuals"]["stationarity_inf"] < 1e-8


def test_near_degenerate_case_keeps_independent_near_duplicate(solve):
    request = near_degenerate_request()
    response = solve(request)
    reference = scipy_slsqp_reference(request)

    assert response["status"] == "optimal"
    np.testing.assert_allclose(response["solution"]["x"], reference.x, atol=5e-7)
    assert response["kkt_residuals"]["primal_inequality_violation"] < 1e-7
    assert response["kkt_residuals"]["stationarity_inf"] < 1e-7
    assert response["kkt_residuals"]["dual_feasibility_violation"] == 0.0
    assert response["kkt_residuals"]["complementarity_max"] < 1e-7
    assert response["active_set"]["solver_active_inequality_names"] == ["near_duplicate_bound"]
    types = event_types(response)
    assert types.count("blocking_constraint_added") >= 1
    assert any(
        event["detail"].get("constraint_name") == "near_duplicate_bound"
        and event["type"] == "blocking_constraint_added"
        for event in response["events"]
    )


def test_supplied_start_triggers_explicit_negative_multiplier_removal(solve):
    response = solve(negative_multiplier_request())
    types = event_types(response)

    assert response["status"] == "optimal"
    np.testing.assert_allclose(response["solution"]["x"], [0.0, 0.0], atol=1e-10)
    assert response["active_set"]["solver_active_inequality_indices"] == [0, 2]
    assert types.count("negative_multiplier_constraint_removed") >= 2
    assert all(
        event["detail"]["multiplier"] < -event["detail"]["dual_tol"]
        for event in response["events"]
        if event["type"] == "negative_multiplier_constraint_removed"
    )
    assert response["events"][0]["type"] == "validate_supplied_initial_point"


def test_rank_deficient_equalities_use_basis_not_blind_inverse(solve):
    response = solve(rank_deficient_equalities_request())

    assert response["status"] == "optimal"
    np.testing.assert_allclose(response["solution"]["x"], [0.5, 0.5], atol=1e-9)
    assert response["active_set"]["equality_basis_indices"] == [1]
    assert response["active_set"]["dependent_equality_indices"] == [0]
    rank_events = [
        event
        for event in response["events"]
        if event["type"] == "equality_rank_detection"
    ]
    assert rank_events and rank_events[0]["detail"]["rank"] == 1
    assert response["kkt_residuals"]["primal_equality_violation"] < 1e-9
    assert response["uncertainties"] == []
    assert any("dependent equalities represented by basis rows" in note for note in response["notes"])
