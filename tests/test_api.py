from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stl_backend.api import create_app
from stl_backend.errors import CalculationError


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "api-runs.sqlite3")))


def test_health_and_fixed_configuration(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    config = client.get("/api/v1/config")
    assert config.status_code == 200
    payload = config.json()
    assert payload["schema_version"] == "decomposition-response-v1"
    assert payload["config"]["period"] == 7
    assert payload["config"]["seasonal_window"] == 7
    assert payload["config"]["trend_window"] == 15
    assert payload["config"]["low_pass_window"] == 9
    assert payload["config"]["minimum_observations"] == 14


def test_decomposition_contract_and_run_events(client, robust_fixture):
    response = client.post(
        "/api/v1/decompositions",
        json={"values": robust_fixture["observed"], "run_id": "api-1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "api-1"
    assert payload["run_number"] == 1
    components = payload["components"]
    observed = payload["observed"]
    assert len(observed) == 140
    assert len(components["trend"]) == 140
    assert len(components["seasonal"]) == 140
    assert len(components["residual"]) == 140
    assert len(payload["robust_weights"]) == 140
    assert payload["diagnostics"]["reconstruction_max_abs_error"] < 1e-12
    assert payload["diagnostics"]["zero_weight_count"] >= 3

    for position in robust_fixture["outlier_positions"]:
        assert payload["robust_weights"][position] < 1e-8

    run_response = client.get("/api/v1/runs/api-1")
    assert run_response.status_code == 200
    assert run_response.json()["status"] == "complete"
    assert run_response.json()["result"]["run_id"] == "api-1"

    events_response = client.get("/api/v1/runs/api-1/events")
    assert events_response.status_code == 200
    events = events_response.json()["events"]
    assert events_response.json()["event_count"] == 18
    assert events[0]["stage"] == "contract_validation"
    assert events[1]["state"]["reconstruction_max_abs_error"] >= 0.0
    assert "judgment" in events[1]


def test_duplicate_run_id_returns_state_conflict(client, robust_fixture):
    first = client.post(
        "/api/v1/decompositions",
        json={"values": robust_fixture["observed"], "run_id": "duplicate"},
    )
    assert first.status_code == 200

    second = client.post(
        "/api/v1/decompositions",
        json={"values": robust_fixture["observed"], "run_id": "duplicate"},
    )
    assert second.status_code == 409
    error = second.json()["error"]
    assert error["code"] == "state_conflict"
    assert error["category"] == "state"


def test_missing_values_are_not_zero_filled(client):
    response = client.post(
        "/api/v1/decompositions",
        content=b'{"run_id":"missing-value","values":[1.0,NaN,'
        + b",".join([b"1.0"] * 12)
        + b"]}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "input_contract_violation"
    assert "missing" in error["message"]
    stored = client.get("/api/v1/runs/missing-value")
    assert stored.status_code == 200
    assert stored.json()["status"] == "failed"
    assert stored.json()["error"]["category"] == "input"


def test_short_series_reports_required_period_boundary(client):
    response = client.post(
        "/api/v1/decompositions",
        json={"run_id": "too-short", "values": [1.0] * 13},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "input_contract_violation"
    assert error["details"]["minimum_observations"] == 14
    stored = client.get("/api/v1/runs/too-short")
    assert stored.status_code == 200
    assert stored.json()["error"]["code"] == "input_contract_violation"


def test_empty_and_non_numeric_schema_errors_are_input_contract(client):
    empty = client.post("/api/v1/decompositions", json={"values": []})
    assert empty.status_code == 400
    assert empty.json()["error"]["code"] == "input_contract_violation"

    non_numeric = client.post(
        "/api/v1/decompositions", json={"values": ["a"] * 14}
    )
    assert non_numeric.status_code == 400
    assert non_numeric.json()["error"]["code"] == "input_contract_violation"


def test_calculation_failure_is_distinct_from_bad_input(client, monkeypatch, robust_fixture):
    from stl_backend import api

    def broken(*args, **kwargs):
        raise CalculationError("simulated numerical failure")

    monkeypatch.setattr(api, "decompose_values", broken)
    response = client.post(
        "/api/v1/decompositions",
        json={"values": robust_fixture["observed"], "run_id": "calc-fail"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "calculation_failed"


def test_unknown_run_is_404(client):
    response = client.get("/api/v1/runs/missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "run_not_found"
