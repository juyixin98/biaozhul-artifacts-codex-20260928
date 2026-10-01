from __future__ import annotations

import json

from fastapi.testclient import TestClient

from entropic_ot.service import create_app


def make_request(**overrides):
    payload = {
        "a": [0.5, 0.5],
        "b": [0.5, 0.5],
        "cost": [[0.0, 1.0], [1.0, 0.0]],
        "epsilon": 1.0,
        "max_iterations": 10,
        "tolerance": 1e-10,
    }
    payload.update(overrides)
    return payload


def test_solve_endpoint_returns_plan_and_separate_objectives(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))

    response = client.post("/solve", json=make_request())

    assert response.status_code == 200
    body = response.json()
    assert body["feasible"] is True
    assert body["converged"] is True
    assert body["stop_reason"] == "converged"
    assert body["original_cost"] > 0.2
    assert body["regularized_cost"] > body["original_cost"]
    assert body["entropy_penalty"] > 0.0
    assert body["diagnostics"]["active_rows"] == [0, 1]
    run_id = body["run_id"]
    log_file = tmp_path / f"{run_id}.jsonl"
    events = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert events[0]["event"] == "start"
    assert events[0]["inputs"]["a"] == make_request()["a"]
    assert events[0]["inputs"]["b"] == make_request()["b"]
    assert events[0]["inputs"]["cost"] == make_request()["cost"]
    assert events[-1]["event"] == "finish"


def test_request_validation_uses_input_quality_category(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))

    response = client.post("/solve", json=make_request(a=[-0.1, 1.1]))

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["category"] == "input_quality"
    assert error["code"] == "RequestValidationError"
    assert "errors" in error["details"]


def test_kernel_quality_error_reports_run_id(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/solve",
        json=make_request(a=[0.5, 0.5], b=[0.4, 0.5]),
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["category"] == "input_quality"
    assert error["run_id"]


def test_budget_expiry_returns_509_with_nonfeasible_best_effort(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))
    payload = make_request(
        a=[0.7, 0.3],
        b=[0.4, 0.6],
        cost=[[0.0, 100.0], [100.0, 0.0]],
        epsilon=1e-8,
        max_iterations=1,
        tolerance=1e-10,
    )

    response = client.post("/solve", json=payload)

    assert response.status_code == 509
    error = response.json()["error"]
    assert error["category"] == "resource_exhausted"
    best_effort = error["details"]["best_effort"]
    assert best_effort["feasible"] is False
    assert best_effort["converged"] is False
    assert best_effort["stop_reason"] == "iteration_budget_exhausted"
    assert best_effort["column_error"] > 0.1
    run_id = best_effort["run_id"]
    events = [json.loads(line) for line in (tmp_path / f"{run_id}.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "failure"
    assert events[-1]["category"] == "resource_exhausted"


def test_idempotent_success_is_replayed_without_new_run(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))
    payload = make_request(idempotency_key="same-success")

    first = client.post("/solve", json=payload)
    second = client.post("/solve", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["run_id"] == first.json()["run_id"]
    assert second.json()["replay"]["replayed"] is True


def test_idempotency_key_reuse_with_changed_payload_is_conflict(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))

    first = client.post("/solve", json=make_request(idempotency_key="k"))
    second = client.post(
        "/solve",
        json=make_request(idempotency_key="k", epsilon=0.2),
    )

    assert first.status_code == 200
    assert second.status_code == 409
    error = second.json()["error"]
    assert error["category"] == "state_conflict"
    assert "existing_request_hash" in error["details"]


def test_computation_failure_has_distinct_category(tmp_path, monkeypatch) -> None:
    from entropic_ot import kernel

    def broken_logsumexp(*args, **kwargs):
        if kwargs.get("axis") == 0:
            raise ArithmeticError("forced")
        return __import__("numpy").array([0.0, 0.0])

    monkeypatch.setattr(kernel, "logsumexp", broken_logsumexp)
    client = TestClient(create_app(tmp_path))

    response = client.post("/solve", json=make_request())

    assert response.status_code == 500
    assert response.json()["error"]["category"] == "computation_failed"
