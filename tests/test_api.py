import json
import os

import pytest
from fastapi.testclient import TestClient

from entropic_ot.service import create_app


def test_health_endpoint():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_solve_endpoint_returns_plan_and_separate_objectives(tmp_path):
    log_path = tmp_path / "events.jsonl"
    old_value = os.environ.get("ENTROPIC_OT_LOG_PATH")
    os.environ["ENTROPIC_OT_LOG_PATH"] = str(log_path)
    try:
        client = TestClient(create_app())
        response = client.post(
            "/v1/optimal-transport",
            json={
                "source_mass": [0.25, 0.75],
                "target_mass": [0.75, 0.25],
                "cost": [[0, 2], [2, 0]],
                "epsilon": 0.5,
                "max_iterations": 1000,
                "atol": 1e-10,
                "rtol": 1e-8,
            },
        )
    finally:
        if old_value is None:
            os.environ.pop("ENTROPIC_OT_LOG_PATH", None)
        else:
            os.environ["ENTROPIC_OT_LOG_PATH"] = old_value

    assert response.status_code == 200
    body = response.json()
    assert len(body["run_id"]) == 32
    assert body["plan"][0][0] == pytest.approx(body["plan"][1][1], abs=5e-9)
    assert body["diagnostics"]["feasible"] is True
    assert body["diagnostics"]["row_marginal_error"] <= 2e-9
    assert body["diagnostics"]["column_marginal_error"] <= 2e-9
    assert body["transport_cost"] != body["regularized_objective"]

    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    events = {line["event"] for line in lines}
    assert {"solve_started", "solve_succeeded"} <= events
    success = next(line for line in lines if line["event"] == "solve_succeeded")
    assert success["run_id"] == body["run_id"]
    assert success["replay"]["cost"] == [[0, 2], [2, 0]]
    assert success["diagnostics"]["iterations"] == body["diagnostics"]["iterations"]


def test_model_parse_failure_is_input_error_with_run_id():
    client = TestClient(create_app())

    response = client.post(
        "/v1/optimal-transport",
        json={
            "source_mass": [1],
            "target_mass": [1],
            "cost": [[0]],
            "epsilon": 0,
            "max_iterations": 10,
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "input_error"
    assert body["run_id"]
    assert body["details"]["validation_issues"]
    assert '"epsilon":0' in body["details"]["raw_body"]


def test_non_finite_tolerance_is_input_error():
    client = TestClient(create_app())

    response = client.post(
        "/v1/optimal-transport",
        json={
            "source_mass": [1],
            "target_mass": [1],
            "cost": [[0]],
            "epsilon": 0.1,
            "atol": "Infinity",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "input_error"
    assert body["details"]["validation_issues"][0]["loc"][-1] == "atol"


def test_semantic_mass_mismatch_is_input_error():
    client = TestClient(create_app())

    response = client.post(
        "/v1/optimal-transport",
        json={
            "source_mass": [0.4, 0.6],
            "target_mass": [0.5, 0.6],
            "cost": [[0, 1], [1, 0]],
            "epsilon": 0.1,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == "input_error"


def test_budget_failure_is_resource_exhausted_and_not_feasible():
    client = TestClient(create_app())

    response = client.post(
        "/v1/optimal-transport",
        json={
            "source_mass": [0.2, 0.3, 0.5],
            "target_mass": [0.1, 0.4, 0.5],
            "cost": [[0, 1, 2], [2, 0, 1], [1, 2, 0]],
            "epsilon": 0.01,
            "max_iterations": 3,
        },
    )

    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "resource_exhausted"
    assert body["diagnostics"]["feasible"] is False
    assert body["diagnostics"]["stopped_reason"] == "iteration_budget_exhausted"
