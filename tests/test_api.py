from __future__ import annotations

import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.diagnostics import FitStatus
from app.main import app


client = TestClient(app)


def exponential_payload(**overrides):
    x = np.linspace(0.0, 2.0, 20)
    y = 2.0 * np.exp(-0.7 * x) + 0.25
    payload = {
        "expression": "a*exp(-b*x)+c",
        "parameter_names": ["a", "b", "c"],
        "x": x.tolist(),
        "y": y.tolist(),
        "initial": [0.2, 2.0, 0.01],
        "bounds": {
            "lower": [0.0, 0.0, 0.0],
            "upper": [10.0, 5.0, 10.0],
        },
        "options": {"check_derivatives": True},
    }
    payload.update(overrides)
    return payload


def test_health_reports_runtime_versions():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"]
    assert payload["numpy"] == np.__version__


def test_fit_endpoint_returns_concrete_parameters_residuals_and_iterations():
    response = client.post(
        "/api/v1/least-squares/fit", json=exponential_payload()
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["status"] in {
        FitStatus.OPTIMAL.value,
        FitStatus.CONVERGED_COST_TOLERANCE.value,
    }
    assert payload["parameters"]["a"] == pytest.approx(2.0, abs=1e-6)
    assert payload["parameters"]["b"] == pytest.approx(0.7, abs=1e-6)
    assert payload["parameters"]["c"] == pytest.approx(0.25, abs=1e-6)
    assert len(payload["residuals"]) == 20
    assert payload["identifiability"]["rank"] == 3
    assert payload["derivative_check"]["passed"] is True
    assert payload["iterations"]
    assert payload["run_id"]
    assert len(payload["input_fingerprint"]) == 16


def test_invalid_expression_is_rejected_without_success(caplog):
    payload = exponential_payload(expression="a*exp(-b*x)+secret")
    with caplog.at_level(logging.WARNING):
        response = client.post("/api/v1/least-squares/fit", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert body["status"] == FitStatus.INVALID_EXPRESSION.value
    assert "secret" in body["message"]
    assert body["run_id"]
    failure_logs = [record for record in caplog.records if record.getMessage() == "fit_failed"]
    assert failure_logs
    assert failure_logs[0].run_id == body["run_id"]


def test_non_convergence_is_200_with_explicit_failure_status():
    payload = exponential_payload(options={"max_iterations": 1})
    response = client.post("/api/v1/least-squares/fit", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["status"] == FitStatus.NON_CONVERGED_MAX_ITERATIONS.value
    assert body["iterations"][0]["actual_reduction"] > 0.0
    assert body["iterations"][0]["predicted_reduction"] > 0.0


def test_illegal_initial_value_has_distinct_http_error():
    x = [0.0, 1.0]
    payload = {
        "expression": "a*log(x)",
        "parameter_names": ["a"],
        "x": x,
        "y": [0.0, 0.0],
        "initial": [1.0],
        "bounds": {"lower": [0.0], "upper": [10.0]},
    }
    response = client.post("/api/v1/least-squares/fit", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["status"] == FitStatus.INVALID_INITIAL_RESIDUAL.value


def test_shape_validation_uses_invalid_input_not_success():
    payload = exponential_payload()
    payload["y"] = payload["y"][:-1]
    response = client.post("/api/v1/least-squares/fit", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert body["status"] == FitStatus.INVALID_INPUT.value


def test_iteration_logs_include_identity_versions_progress_and_judgment(caplog):
    with caplog.at_level(logging.INFO):
        response = client.post(
            "/api/v1/least-squares/fit", json=exponential_payload()
        )
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    fingerprint = response.json()["input_fingerprint"]
    iteration_logs = [record for record in caplog.records if record.getMessage() == "iteration"]
    assert iteration_logs
    for record in iteration_logs:
        assert record.run_id == run_id
        assert record.input_fingerprint == fingerprint
        assert record.service_version
        assert record.numpy_version
        assert record.scipy_version
        assert "/" in record.progress
        assert record.actual_reduction is not None
        assert record.predicted_reduction is not None
        assert record.criterion
