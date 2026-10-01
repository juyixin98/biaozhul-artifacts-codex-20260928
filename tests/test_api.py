from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
import threading

from consensus_admm.api import create_app
from consensus_admm.config import ServerSettings


def make_client(tmp_path, **settings_overrides):
    settings = ServerSettings(
        data_dir=tmp_path,
        max_runs=settings_overrides.get("max_runs", 8),
        max_problem_dim=settings_overrides.get("max_problem_dim", 16),
        max_iterations=settings_overrides.get("max_iterations", 200),
    )
    return TestClient(create_app(settings))


def problem_payload(**overrides):
    payload = {
        "name": "asymmetric-two-process",
        "dimension": 3,
        "workers": [
            {
                "id": "worker-a",
                "P": [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]],
                "q": [1.0, -2.0, 0.5],
                "sample_count": 11,
            },
            {
                "id": "worker-b",
                "P": [[4.0, 0.2, 0.0], [0.2, 5.0, -0.1], [0.0, -0.1, 2.0]],
                "q": [-3.0, 1.0, 4.0],
                "r": 0.25,
                "sample_count": 83,
            },
        ],
        "initial_rho": 1.0,
        "eps_abs": 1e-8,
        "eps_rel": 1e-8,
        "max_iterations": 120,
        "deadline_ms": 1000,
    }
    payload.update(overrides)
    return payload


def create_run(client, **overrides):
    response = client.post("/runs", json=problem_payload(**overrides))
    assert response.status_code == 201, response.text
    body = response.json()
    return body["run_id"], body


def independent_local_submission(client, run_id, worker_id, task):
    run = client.get(f"/runs/{run_id}").json()
    worker = next(
        item
        for item in run["problem"]["workers"]
        if item["id"] == worker_id
    )
    P = np.asarray(worker["P"], dtype=float)
    q = np.asarray(worker["q"], dtype=float)
    rho = float(task["rho"])
    z = np.asarray(task["z"], dtype=float)
    u = np.asarray(task["u"], dtype=float)
    x = np.linalg.solve(P + rho * np.eye(P.shape[0]), rho * z - u - q)
    objective = float(0.5 * x @ P @ x + q @ x + float(worker.get("r", 0.0)))
    return {
        "round_token": task["round_token"],
        "rho": rho,
        "z_pre": z.tolist(),
        "u_pre": u.tolist(),
        "x": x.tolist(),
        "objective": objective,
        "stationarity_residual": float(
            np.linalg.norm(P @ x + q + u + rho * (x - z))
        ),
        "solve_seconds": 0.0,
    }


def post_independent_submission(client, run_id, worker_id):
    task = client.get(f"/runs/{run_id}/workers/{worker_id}/task").json()
    payload = independent_local_submission(client, run_id, worker_id, task)
    response = client.post(
        f"/runs/{run_id}/workers/{worker_id}/submissions",
        json=payload,
    )
    assert response.status_code == 200, response.text
    return response.json(), payload


def run_to_terminal(client, run_id):
    iterations = 0
    while True:
        run = client.get(f"/runs/{run_id}").json()
        if run["status"] in {"converged", "max_iterations_reached"}:
            return run
        assert run["status"] == "running"
        for worker_id in ("worker-a", "worker-b"):
            post_independent_submission(client, run_id, worker_id)
        iterations += 1
        assert iterations <= 120


def test_full_run_converges_to_independent_central_solution(tmp_path):
    client = make_client(tmp_path)
    run_id, run = create_run(client)
    assert run["status"] == "running"
    assert run["iteration"] == 0
    assert run["current_round"]["missing_workers"] == ["worker-a", "worker-b"]

    final = run_to_terminal(client, run_id)
    workers = final["problem"]["workers"]
    P = sum(np.asarray(worker["P"], dtype=float) for worker in workers)
    q = sum(np.asarray(worker["q"], dtype=float) for worker in workers)
    x_star = np.linalg.solve(P, -q)

    assert final["status"] == "converged"
    np.testing.assert_allclose(np.asarray(final["z"]), x_star, atol=1e-7)
    assert final["history"][-1]["primal_ok"] is True
    assert final["history"][-1]["dual_ok"] is True
    assert len({item["iteration"] for item in final["history"]}) == len(final["history"])

    events = client.get(f"/runs/{run_id}/events").json()["events"]
    assert events[0]["type"] == "run_created"
    assert events[0]["data"]["run_id"] == run_id
    assert events[1]["type"] == "round_opened"
    assert all(event["run_id"] == run_id for event in events)


def test_timeout_holds_round_then_resume_uses_same_round_data(tmp_path):
    client = make_client(tmp_path)
    run_id, _ = create_run(client, deadline_ms=30)

    run, first_payload = post_independent_submission(client, run_id, "worker-a")
    assert run["current_round"]["submitted_workers"] == ["worker-a"]
    time.sleep(0.05)

    late = client.post(
        f"/runs/{run_id}/workers/worker-b/submissions",
        json=independent_local_submission(
            client,
            run_id,
            "worker-b",
            client.get(f"/runs/{run_id}/workers/worker-b/task").json(),
        ),
    )
    assert late.status_code == 409
    assert late.json()["error"]["category"] == "state_conflict"
    assert late.json()["error"]["code"] == "round_deadline_expired"

    timed_out = client.get(f"/runs/{run_id}").json()
    assert timed_out["status"] == "timed_out"
    assert timed_out["current_round"]["iteration"] == 0
    assert timed_out["current_round"]["missing_workers"] == ["worker-b"]

    before_timeout = client.post(f"/runs/{run_id}/timeout", json={"reason": "x"})
    assert before_timeout.status_code == 409
    resume = client.post(
        f"/runs/{run_id}/resume",
        json={"deadline_ms": 500, "reason": "test-retry"},
    )
    assert resume.status_code == 200
    resumed = resume.json()
    assert resumed["status"] == "running"
    assert resumed["iteration"] == 0
    assert resumed["current_round"]["round_token"] == timed_out["current_round"]["round_token"]
    assert resumed["current_round"]["submitted_workers"] == ["worker-a"]

    # A stale token from another round is a computation-contract failure.
    task_b = client.get(f"/runs/{run_id}/workers/worker-b/task").json()
    payload_b = independent_local_submission(client, run_id, "worker-b", task_b)
    payload_b["round_token"] = "wrong-token"
    wrong = client.post(
        "/runs/{}/workers/worker-b/submissions".format(run_id),
        json=payload_b,
    )
    assert wrong.status_code == 422
    assert wrong.json()["error"]["category"] == "computation_failure"

    payload_b = independent_local_submission(client, run_id, "worker-b", task_b)
    completed = client.post(
        f"/runs/{run_id}/workers/worker-b/submissions",
        json=payload_b,
    )
    assert completed.status_code == 200
    body = completed.json()
    assert body["iteration"] == 1

    duplicate = client.post(
        f"/runs/{run_id}/workers/worker-a/submissions",
        json=first_payload,
    )
    assert duplicate.status_code == 422
    assert duplicate.json()["error"]["code"] == "round_token_mismatch"


def test_rho_adjustment_rescales_dual_at_round_boundary(tmp_path):
    client = make_client(tmp_path)
    run_id, _ = create_run(client, rho_schedule=[{"iteration": 1, "rho": 3.5}])
    for _ in range(2):
        run = client.get(f"/runs/{run_id}").json()
        for worker_id in ("worker-a", "worker-b"):
            post_independent_submission(client, run_id, worker_id)

    run = client.get(f"/runs/{run_id}").json()
    opened_round_one = [
        event
        for event in client.get(f"/runs/{run_id}/events").json()["events"]
        if event["type"] == "round_opened" and event["data"]["iteration"] == 1
    ][0]
    first = run["history"][0]
    assert opened_round_one["data"]["rho"] == 3.5
    assert opened_round_one["data"]["rho_previous"] == 1.0
    for worker_id in ("worker-a", "worker-b"):
        old_y = 1.0 * np.asarray(first["u"][worker_id])
        new_y = 3.5 * np.asarray(opened_round_one["data"]["u_pre"][worker_id])
        np.testing.assert_allclose(new_y, old_y, atol=1e-12)


def test_concurrent_timeout_and_resume_cannot_corrupt_round(tmp_path):
    client = make_client(tmp_path)
    run_id, _ = create_run(client, deadline_ms=100)
    post_independent_submission(client, run_id, "worker-a")
    time.sleep(0.12)
    outcomes = []

    def mark_timeout():
        response = client.post(
            f"/runs/{run_id}/timeout",
            json={"reason": "concurrent-timeout"},
        )
        outcomes.append(("timeout", response.status_code))

    def try_resume():
        time.sleep(0.003)
        response = client.post(
            f"/runs/{run_id}/resume",
            json={"deadline_ms": 500, "reason": "concurrent-resume"},
        )
        outcomes.append(("resume", response.status_code))

    threads = [threading.Thread(target=mark_timeout), threading.Thread(target=try_resume)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    statuses = dict(outcomes)
    assert statuses["timeout"] == 200
    assert statuses["resume"] in {200, 409}
    run = client.get(f"/runs/{run_id}").json()
    if statuses["resume"] == 200:
        assert run["status"] == "running"
    else:
        assert run["status"] == "timed_out"
    assert run["current_round"]["submitted_workers"] == ["worker-a"]


def test_restart_from_disk_continues_without_mixing_rounds(tmp_path):
    client = make_client(tmp_path)
    run_id, _ = create_run(client)
    for worker_id in ("worker-a", "worker-b"):
        post_independent_submission(client, run_id, worker_id)
    run = client.get(f"/runs/{run_id}").json()
    assert run["iteration"] == 1

    restarted_client = make_client(tmp_path)
    recovered = restarted_client.get(f"/runs/{run_id}").json()
    assert recovered["iteration"] == 1
    assert len(recovered["history"]) == 1
    final = run_to_terminal(restarted_client, run_id)
    assert final["status"] == "converged"


def test_max_iterations_reports_which_tolerance_was_not_reached(tmp_path):
    client = make_client(tmp_path)
    run_id, _ = create_run(
        client,
        max_iterations=1,
        eps_abs=1e-12,
        eps_rel=1e-12,
        deadline_ms=1000,
    )
    for worker_id in ("worker-a", "worker-b"):
        post_independent_submission(client, run_id, worker_id)
    final = client.get(f"/runs/{run_id}").json()
    assert final["status"] == "max_iterations_reached"
    failure = final["failure_classification"]
    assert failure["category"] == "tolerance_not_reached"
    assert set(failure["missing_conditions"]).issubset({"primal", "dual"})
    assert failure["missing_conditions"]


def test_error_categories_are_distinct(tmp_path):
    client = make_client(tmp_path, max_problem_dim=2)
    too_large = client.post("/runs", json=problem_payload(dimension=3))
    assert too_large.status_code == 507
    assert too_large.json()["error"]["category"] == "resource_exhausted"

    bad = client.post(
        "/runs",
        json={
            "name": "bad-curvature",
            "dimension": 1,
            "workers": [
                {"id": "worker-a", "P": [[-1.0]], "q": [0.0]},
                {"id": "worker-b", "P": [[1.0]], "q": [0.0]},
            ],
        },
    )
    assert bad.status_code == 400
    assert bad.json()["error"]["category"] == "input_error"

    missing = client.get("/runs/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["error"]["category"] == "input_error"


def test_wrong_objective_is_computation_failure_and_evidence_is_recorded(tmp_path):
    client = make_client(tmp_path)
    run_id, _ = create_run(client)
    task = client.get("/runs/{}/workers/worker-a/task".format(run_id)).json()
    payload = independent_local_submission(client, run_id, "worker-a", task)
    payload["objective"] += 10.0
    response = client.post(
        f"/runs/{run_id}/workers/worker-a/submissions",
        json=payload,
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "objective_mismatch"
    events = client.get(f"/runs/{run_id}/events").json()["events"]
    assert events[-1]["type"] == "submission_rejected"
    run = client.get(f"/runs/{run_id}").json()
    assert run["current_round"]["submitted_workers"] == []
