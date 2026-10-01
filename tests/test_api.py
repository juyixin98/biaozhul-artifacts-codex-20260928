from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient


def make_payload(run_id: str = "api-run") -> dict:
    x = np.linspace(0.0, 2.0, 12)
    y = 2.0 * np.exp(-1.5 * x) + 0.3
    return {
        "expression": "amp*exp(-rate*x)+floor",
        "run_id": run_id,
        "parameters": {
            "amp": {"initial": 0.1, "lower": 0.0, "upper": 10.0, "scale": 2.0},
            "rate": {"initial": 0.01, "lower": 0.0, "upper": 10.0, "scale": 1.5},
            "floor": {"initial": 1.5, "lower": 0.0, "upper": 10.0, "scale": 1.0},
        },
        "x": x.tolist(),
        "y": y.tolist(),
        "options": {"max_iterations": 80, "tolerance": 1e-4},
    }


def test_health_and_solve_success(monkeypatch, tmp_path):
    monkeypatch.setenv("NLS_LOG_PATH", str(tmp_path / "api.jsonl"))
    from app.main import app

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.post("/solve", json=make_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"] == "CONVERGED_INTERIOR"
    assert data["run_id"] == "api-run"
    assert data["parameters"]["amp"] == 1.9999999999996223 or abs(data["parameters"]["amp"] - 2.0) < 1e-5
    assert abs(data["parameters"]["rate"] - 1.5) < 1e-5
    assert abs(data["parameters"]["floor"] - 0.3) < 1e-5
    assert data["diagnostics"]["finite_difference_max_error"] < 1e-7
    events = [json.loads(line) for line in (tmp_path / "api.jsonl").read_text().splitlines()]
    assert {event["run_id"] for event in events} == {"api-run"}
    assert events[0]["versions"]["scipy"]


def test_solver_level_failure_is_structured_not_500(monkeypatch, tmp_path):
    monkeypatch.setenv("NLS_LOG_PATH", str(tmp_path / "api-fail.jsonl"))
    from app.main import app

    payload = make_payload("api-fail")
    payload["expression"] = "amp*sqrt(x-offset)+floor"
    payload["parameters"] = {
        "amp": {"initial": 1.0, "lower": 0.0, "upper": 3.0},
        "offset": {"initial": 1.5, "lower": 0.0, "upper": 3.0},
        "floor": {"initial": 0.0, "lower": 0.0, "upper": 3.0},
    }
    payload["options"]["derivative_check"] = False

    with TestClient(app) as client:
        response = client.post("/solve", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["status"] == "ILLEGAL_FUNCTION_VALUE"
    assert "Illegal value" in data["reason"]


def test_invalid_bounds_are_400(monkeypatch, tmp_path):
    monkeypatch.setenv("NLS_LOG_PATH", str(tmp_path / "bad-request.jsonl"))
    from app.main import app

    payload = make_payload("bad-request")
    payload["parameters"]["amp"]["lower"] = 11.0

    with TestClient(app) as client:
        response = client.post("/solve", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "value_error"


def test_unsafe_expression_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("NLS_LOG_PATH", str(tmp_path / "unsafe.jsonl"))
    from app.main import app

    payload = make_payload("unsafe")
    payload["expression"] = "__import__('os')"

    with TestClient(app) as client:
        response = client.post("/solve", json=payload)

    assert response.status_code == 400
    assert response.json()["status"] == "INVALID_REQUEST"
