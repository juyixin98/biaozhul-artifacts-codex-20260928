import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.models import FeasibilityRequest, SolveRequest
from conftest import (
    boundary_request,
    constraint,
    infeasible_request,
    scipy_highs_feasibility_reference,
)


def test_phase_one_is_separate_and_reports_infeasible_category(solve):
    request = infeasible_request()
    response = solve(request)
    reference = scipy_highs_feasibility_reference(request)

    assert reference.status == 2
    assert response["status"] == "infeasible"
    assert response["failure_category"] == "infeasible"
    assert response["solution"] is None
    assert response["events"][0]["phase"] == "phase_1"
    assert any(event["type"] == "feasibility_lp_infeasible" for event in response["events"])
    assert "HiGHS" in response["reason"]


def test_feasibility_endpoint_uses_independent_lp():
    client = TestClient(app)
    payload = FeasibilityRequest(
        request_id="phase-one-only",
        n=2,
        constraints=[
            constraint([1, 1], "<=", 1),
            constraint([1, 0], ">=", 0),
            constraint([0, 1], ">=", 0),
        ],
    ).model_dump()

    result = client.post("/api/v1/qp/feasibility", json=payload)
    assert result.status_code == 200
    body = result.json()
    assert body["request_id"] == "phase-one-only"
    assert body["status"] == "feasible"
    x = np.asarray(body["solution"]["x"])
    assert x.sum() <= 1.0 + 1e-8
    assert np.all(x >= -1e-8)
    assert body["max_primal_violation"] < 1e-8
    assert "solve_feasibility_lp" in [event["type"] for event in body["events"]]


def test_non_positive_definite_hessian_has_specific_failure(solve):
    request = SolveRequest(
        objective={"quadratic": [[1.0, 0.0], [0.0, -1.0]], "linear": [0.0, 0.0]},
        constraints=[],
    )
    try:
        solve(request)
    except Exception as exc:
        assert exc.code == "hessian_not_positive_definite"
    else:
        raise AssertionError("Non-positive-definite Hessian must be rejected")


def test_request_dimension_mismatch_is_validation_failure():
    client = TestClient(app)
    response = client.post(
        "/api/v1/qp/solve",
        json={
            "objective": {"quadratic": [[1.0, 0.0], [0.0, 1.0]], "linear": [0.0]},
            "constraints": [],
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["failure_category"] == "linear_dimension_mismatch"
    assert body["request_id"]
    assert len(body["failure_reasons"]) == 1


def test_semantic_error_preserves_caller_request_id():
    client = TestClient(app)
    response = client.post(
        "/api/v1/qp/solve",
        json={
            "request_id": "semantic-error-id",
            "objective": {"quadratic": [[1.0, 0.0], [0.0, 1.0]], "linear": [0.0]},
            "constraints": [],
        },
    )
    body = response.json()
    assert response.status_code == 400
    assert body["request_id"] == "semantic-error-id"
    assert body["failure_category"] == "linear_dimension_mismatch"


def test_response_contains_identity_trace_and_runtime_versions(solve):
    response = solve(boundary_request(request_id="trace-id-123"))

    assert response["request_id"] == "trace-id-123"
    assert response["request_fingerprint_sha256"]
    assert response["runtime"]["service_version"]
    assert response["runtime"]["numpy"]
    assert response["runtime"]["scipy"]
    assert response["runtime"]["processing_location"]
    assert response["events"]
    assert response["tolerances"]["rank"] == 1e-11
