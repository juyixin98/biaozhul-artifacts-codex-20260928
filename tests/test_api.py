from fastapi.testclient import TestClient

from lp_service.app import create_app

from .fixtures import two_variable_optimal


client = TestClient(create_app())


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_solve_endpoint_returns_evidence_and_request_id():
    response = client.post(
        "/solve",
        json=two_variable_optimal().model_dump(),
        headers={"x-request-id": "api-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "api-1"
    assert body["status"] == "optimal"
    assert body["objective_value"] == 2.0
    assert body["diagnostic"]["checks"]["dual_stationarity"] is True


def test_invalid_dimension_is_rejected_with_specific_diagnostic():
    response = client.post(
        "/solve",
        json={
            "objective": [1.0, 2.0],
            "constraints": [{"sense": "eq", "rhs": 1.0, "coefficients": [1.0]}],
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "invalid_request"
    assert "length must equal" in body["diagnostic"]["message"]
