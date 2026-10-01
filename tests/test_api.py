from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _exponential_payload(**config_updates):
    rng = np.random.default_rng(304)
    x = np.linspace(0.0, 4.0, 30)
    y = 3.0 * np.exp(-0.7 * x) + 0.5 + rng.normal(0.0, 0.01, x.size)
    config = {"max_iterations": 100, "gradient_tol": 1.0e-7}
    config.update(config_updates)
    return {
        "expression": "a*exp(-b*x)+c",
        "x": x.tolist(),
        "y": y.tolist(),
        "parameters": [
            {"name": "a", "initial": 0.1, "lower": 0.0, "upper": 10.0, "scale": 1.0},
            {"name": "b", "initial": 2.5, "lower": 0.0, "upper": 5.0, "scale": 0.2},
            {"name": "c", "initial": 2.0, "lower": -2.0, "upper": 5.0, "scale": 1.0},
        ],
        "config": config,
    }


def test_health_includes_run_identity():
    response = client.get("/health", headers={"X-Run-ID": "health-304"})
    assert response.status_code == 200
    assert response.json()["run_id"] == "health-304"


def test_solve_success_reports_explicit_convergence_and_logs():
    response = client.post("/solve", json=_exponential_payload(), headers={"X-Run-ID": "api-success-304"})
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "api-success-304"
    assert body["status"] == "converged"
    assert body["stationarity_type"] == "interior_stationary"
    assert body["identifiability"]["identifiable"] is True
    assert body["log_tail"]
    assert any(event["run_id"] == "api-success-304" and event["stage"] == "iteration" for event in body["log_tail"])
    assert all("numpy_version" in event for event in body["log_tail"] if event["stage"] == "run_start")


def test_invalid_expression_is_400_not_success():
    payload = _exponential_payload()
    payload["expression"] = "open('x').read()"
    response = client.post("/solve", json=payload, headers={"X-Run-ID": "api-invalid-304"})
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "invalid_expression"
    assert body["run_id"] == "api-invalid-304"
    assert body["log_tail"]


def test_nonconvergence_is_422_with_explicit_failure_category():
    response = client.post(
        "/solve",
        json=_exponential_payload(max_iterations=1),
        headers={"X-Run-ID": "api-nonconv-304"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "nonconvergent"
    assert body["message"]
    assert body["run_id"] == "api-nonconv-304"


def test_unidentifiable_payload_returns_distinct_failure():
    x = np.linspace(0.0, 1.0, 8)
    payload = {
        "expression": "a+b+c*x",
        "x": x.tolist(),
        "y": (2.0 + 0.5 * x).tolist(),
        "parameters": [
            {"name": name, "initial": 0.0, "lower": -10.0, "upper": 10.0}
            for name in ("a", "b", "c")
        ],
    }
    response = client.post("/solve", json=payload, headers={"X-Run-ID": "api-rank-304"})
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "unidentifiable"
    assert body["details"]["identifiability"]["numerical_rank"] == 2


def test_illegal_function_value_has_separate_status():
    x = np.linspace(0.1, 1.0, 8)
    payload = {
        "expression": "log(x-b)",
        "x": x.tolist(),
        "y": np.zeros_like(x).tolist(),
        "parameters": [{"name": "b", "initial": 2.0, "lower": -10.0, "upper": 10.0}],
    }
    response = client.post("/solve", json=payload, headers={"X-Run-ID": "api-illegal-304"})
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "illegal_function_value"
    assert body["details"]["nonfinite_count"] == x.size
