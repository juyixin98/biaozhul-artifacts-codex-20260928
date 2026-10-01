from __future__ import annotations

import time
import threading

from tests.conftest import wait_for_run


def test_version_reports_actual_dependencies(client):
    response = client.get("/version")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "service",
        "python",
        "numpy",
        "scipy",
        "fastapi",
        "pydantic",
        "sqlite",
    }
    assert all(value for value in payload.values())


def test_create_exact_run_returns_concrete_result_and_audit_events(client):
    response = client.post(
        "/api/v1/runs",
        json={"values": [1.0, 2.0, 3.0], "group1_size": 1, "method": "exact"},
    )
    assert response.status_code == 202
    created = response.json()
    run_id = created["run_id"]
    assert created["status"] in {"pending", "running", "completed"}

    run = wait_for_run(client, run_id)
    result = run["result"]

    assert run["status"] == "completed"
    assert run["error"] is None
    assert result["method_used"] == "exact"
    assert result["statistic"] == 1.5
    assert result["p_value"] == 2 / 3
    assert result["total_possible_assignments"] == 3
    assert result["evaluated_permutations"] == 3
    assert result["alternative"] == "two-sided"
    assert "runtime_versions" in result

    events = client.get(f"/api/v1/runs/{run_id}/events").json()
    messages = "\n".join(event["message"] for event in events)
    assert run["input_hash"] in messages
    assert "computation started" in messages
    assert "exact-enumeration: 3/3 completed" in messages
    assert "p_value=0.666666666667" in messages
    assert "decision=not_significant" in messages


def test_api_seed_replay_returns_same_result(client):
    payload = {
        "values": list(range(10)),
        "group1_size": 5,
        "method": "monte_carlo",
        "monte_carlo_replications": 77,
        "random_seed": 314159,
    }
    first_response = client.post("/api/v1/runs", json=payload)
    second_response = client.post("/api/v1/runs", json=payload)
    first_id = first_response.json()["run_id"]
    second_id = second_response.json()["run_id"]

    first = wait_for_run(client, first_id)
    second = wait_for_run(client, second_id)

    first_result = dict(first["result"])
    second_result = dict(second["result"])
    first_result.pop("elapsed_ms")
    second_result.pop("elapsed_ms")
    assert first_result == second_result
    assert first["input_hash"] == second["input_hash"]
    assert first["result"]["random_seed"] == 314159
    assert first["result"]["evaluated_permutations"] == 77
    assert first["result"]["standard_error"] > 0


def test_api_invalid_input_is_explicit_failure_not_success(client):
    response = client.post(
        "/api/v1/runs",
        json={"values": [1.0, 2.0], "group1_size": 2, "method": "exact"},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "INVALID_INPUT"
    assert error["category"] == "invalid_input"
    assert "group1_size" in error["message"]


def test_exact_cap_is_rejected_synchronously(client):
    response = client.post(
        "/api/v1/runs",
        json={
            "values": list(range(30)),
            "group1_size": 15,
            "method": "exact",
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "EXACT_LIMIT_EXCEEDED"
    assert error["details"]["total_possible_assignments"] == 155_117_520


def test_schema_validation_rejects_unknown_or_nonfinite_payloads(client):
    response = client.post(
        "/api/v1/runs",
        json={
            "values": [1.0, "2.0"],
            "group1_size": 1,
            "unexpected": True,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"

    response = client.post(
        "/api/v1/runs",
        json={"values": [1.0, "not-a-number"], "group1_size": 1},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_canceled_run_is_terminal_and_has_no_significance_result(client):
    response = client.post(
        "/api/v1/runs",
        json={
            "values": list(range(28)),
            "group1_size": 14,
            "method": "exact",
        },
    )
    run_id = response.json()["run_id"]
    cancel_response = client.post(f"/api/v1/runs/{run_id}/cancel")
    assert cancel_response.status_code in {200, 202}

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        run_response = client.get(f"/api/v1/runs/{run_id}")
        run = run_response.json()
        if run["status"] == "canceled":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("run was not canceled")

    assert run_response.status_code == 409
    assert run["status"] == "canceled"
    assert run["result"] is None
    assert run["error"]["code"] == "CANCELED"
    assert run["error"]["category"] == "cancellation"

    events = client.get(f"/api/v1/runs/{run_id}/events").json()
    phases = [event["phase"] for event in events]
    assert "cancel-requested" in phases
    assert "canceled" in phases


def test_cancel_during_completion_window_does_not_become_success(client):
    app = client.app
    store = app.state.store
    runner = app.state.runner
    patch_ready = threading.Event()
    release_complete = threading.Event()
    created_run_id = None
    original_complete = store.complete_run
    original_create_run = runner.create_run

    def install_race_monitor(run_input):
        run = original_create_run(run_input)

        def race_cancel_then_complete(actual_run_id, result):
            patch_ready.set()
            release_complete.wait(5)
            store.request_cancel(actual_run_id)
            return original_complete(actual_run_id, result)

        store.complete_run = race_cancel_then_complete
        return run

    runner.create_run = install_race_monitor
    try:
        response = client.post(
            "/api/v1/runs",
            json={"values": [1.0, 2.0, 3.0], "group1_size": 1},
        )
        run_id = response.json()["run_id"]
        created_run_id = run_id
        assert patch_ready.wait(5)
        release_complete.set()
    finally:
        runner.create_run = original_create_run
        if created_run_id is None:
            store.complete_run = original_complete
        release_complete.set()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        run = client.get(f"/api/v1/runs/{run_id}").json()
        if run["status"] in {"completed", "failed", "canceled"}:
            break
        time.sleep(0.01)

    store.complete_run = original_complete

    assert run["status"] == "canceled"
    assert run["result"] is None
    assert run["error"]["code"] == "CANCELED"


def test_missing_run_and_events_have_not_found_errors(client):
    response = client.get("/api/v1/runs/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RUN_NOT_FOUND"

    response = client.get("/api/v1/runs/does-not-exist/events")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RUN_NOT_FOUND"
