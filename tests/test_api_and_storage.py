from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.errors import ComputationError
from app.kernel import stl as stl_module
from app.main import app, get_repository
from app.storage import RunRepository


@pytest.fixture()
def client(tmp_path):
    repository = RunRepository(tmp_path / "test-runs.db")
    app.dependency_overrides[get_repository] = lambda: repository
    with TestClient(app) as test_client:
        yield test_client, repository
    app.dependency_overrides.clear()


def test_decompose_success_persists_components_and_key_trace(
    client, primary_fixture
):
    test_client, repository = client
    payload = {"run_id": "case-primary-001", "values": primary_fixture["values"]}

    response = test_client.post("/decompose", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["run_id"] == "case-primary-001"
    assert len(body["trend"]) == 70
    assert len(body["trace"]) == 17
    assert body["diagnostics"]["additive_max_abs_error"] == pytest.approx(
        0.0, abs=1e-12
    )

    stored = repository.get_run("case-primary-001")
    assert stored["status"] == "success"
    assert stored["config_fingerprint"] == body["diagnostics"]["config_fingerprint"]
    result = json.loads(stored["result_json"])
    assert result["weights"][9] == 0.0
    assert len(stored["trace"]) == 17
    assert stored["trace"][0]["state"]["observed"]["length"] == 70
    assert stored["trace"][-1]["judgment"]["zero_weight_count"] >= 3


def test_missing_value_contract_is_input_error_before_kernel(client):
    test_client, repository = client
    values = [1.0] * 14
    values[2] = float("nan")

    response = test_client.post(
        "/decompose",
        content=json.dumps({"values": values}, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "INPUT_ERROR"
    assert "NaN or infinite" in body["message"]


def test_short_series_remains_input_error_with_replay_details(
    client, primary_fixture
):
    test_client, repository = client
    payload = {"run_id": "case-too-short", "values": primary_fixture["values"][:13]}

    response = test_client.post("/decompose", json=payload)

    assert response.status_code == 400
    assert response.json()["error"] == "INPUT_ERROR"
    stored = repository.get_run("case-too-short")
    assert stored["status"] == "failure"
    assert stored["error_category"] == "INPUT_ERROR"
    assert json.loads(stored["request_json"])["values"] == payload["values"]


def test_duplicate_run_id_is_distinct_state_conflict(client, primary_fixture):
    test_client, _ = client
    payload = {"run_id": "case-duplicate", "values": primary_fixture["values"]}

    assert test_client.post("/decompose", json=payload).status_code == 201
    response = test_client.post("/decompose", json=payload)

    assert response.status_code == 409
    assert response.json()["error"] == "STATE_CONFLICT"


def test_resource_exhaustion_is_distinct(client):
    test_client, repository = client
    payload = {"run_id": "case-too-large", "values": [1.0] * 5001}

    response = test_client.post("/decompose", json=payload)

    assert response.status_code == 413
    assert response.json()["error"] == "RESOURCE_EXHAUSTED"
    stored = repository.get_run("case-too-large")
    assert stored["status"] == "failure"
    assert stored["error_category"] == "RESOURCE_EXHAUSTED"


def test_computation_failure_is_logged_without_being_misclassified(
    client, primary_fixture, monkeypatch
):
    test_client, repository = client

    def broken_smooth(*args, **kwargs):
        raise ComputationError("synthetic smoothing failure", {"stage": "unit-test"})

    monkeypatch.setattr(stl_module, "loess_smooth", broken_smooth)
    response = test_client.post(
        "/decompose",
        json={"run_id": "case-computation", "values": primary_fixture["values"]},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "COMPUTATION_FAILED"
    assert response.json()["details"]["stage"] == "unit-test"
    stored = repository.get_run("case-computation")
    assert stored["status"] == "failure"
    assert stored["error_category"] == "COMPUTATION_FAILED"


def test_storage_failure_remains_storage_failure(client, primary_fixture, monkeypatch):
    test_client, repository = client

    def broken_save_success(*args, **kwargs):
        raise sqlite3.OperationalError("synthetic disk failure")

    monkeypatch.setattr(repository, "save_success", broken_save_success)
    # The low-level storage exception is wrapped by the real public method.
    from app import storage as storage_module

    def raising(*args, **kwargs):
        raise storage_module.StorageError(
            "synthetic disk failure", {"reason": "unit-test"}
        )

    monkeypatch.setattr(repository, "save_success", raising)
    response = test_client.post(
        "/decompose",
        json={"run_id": "case-storage", "values": primary_fixture["values"]},
    )

    assert response.status_code == 500
    assert response.json()["error"] == "STORAGE_FAILED"


def test_version_endpoint_exposes_fixed_specification(client):
    test_client, _ = client
    response = test_client.get("/version")

    assert response.status_code == 200
    body = response.json()
    assert body["spec_version"] == "1.0.0"
    assert body["config"]["period"] == 7
    assert body["config"]["seasonal_window"] == 7
    assert body["config"]["trend_window"] == 15
    assert body["config"]["outer_robust_iterations"] == 5


def test_unknown_route_is_input_error(client):
    test_client, _ = client
    response = test_client.get("/runs/does-not-exist")
    assert response.status_code == 400
    assert response.json()["error"] == "INPUT_ERROR"
