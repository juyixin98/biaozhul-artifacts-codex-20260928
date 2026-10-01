from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
from fastapi.testclient import TestClient

from admm_consensus.kernel import solve_local_update
from admm_consensus.models import QuadraticModel


def payload(*, rho=5.0, max_iterations=80, adaptive_rounds=None):
    rng = np.random.default_rng(91)
    A1 = rng.normal(size=(4, 2))
    A2 = rng.normal(size=(9, 2))
    b1 = rng.normal(size=4)
    b2 = rng.normal(size=9)
    data = {
        "dimension": 2,
        "workers": [
            {"worker_id": 1, "Q": (A1.T @ A1).tolist(), "q": (-A1.T @ b1).tolist()},
            {"worker_id": 2, "Q": (A2.T @ A2).tolist(), "q": (-A2.T @ b2).tolist()},
        ],
        "config": {
            "rho": rho,
            "max_iterations": max_iterations,
            "tolerances": {"primal_abs": 1e-8, "dual_abs": 1e-8},
        },
    }
    if adaptive_rounds is not None:
        data["config"]["adaptive_rho"] = {
            "mu": 2.0,
            "tau_incr": 2.0,
            "tau_decr": 2.0,
            "change_rounds": adaptive_rounds,
        }
    return data


def local_update(client, run_id, worker_id):
    run = client.get(f"/runs/{run_id}").json()["run"]
    task = client.get(f"/runs/{run_id}/task").json()["task"]
    item = next(w for w in task["workers"] if w["worker_id"] == worker_id)
    model = QuadraticModel(Q=np.asarray(item["Q"]), q=np.asarray(item["q"]))
    x = solve_local_update(
        model,
        z=np.asarray(item["z"]),
        u=np.asarray(item["u"]),
        rho=task["rho"],
        worker_id=worker_id,
    )
    response = client.post(
        f"/runs/{run_id}/submit",
        json={"worker_id": worker_id, "round_number": task["round_number"], "rho": task["rho"], "x": x.tolist()},
    )
    response.raise_for_status()
    return response


def test_http_create_auto_round_convergence_and_events(isolated_app):
    client = TestClient(isolated_app.app)
    created = client.post("/runs", json=payload())
    assert created.status_code == 201
    run_id = created.json()["run"]["run_id"]

    response = client.post(f"/runs/{run_id}/auto-round")
    assert response.status_code == 200
    run = response.json()["run"]
    assert run["round_number"] == 2
    assert run["trajectory"][0]["phase_order"] == ["local_updates", "consensus", "dual", "residuals", "termination"]

    while run["status"] == "running":
        run = client.post(f"/runs/{run_id}/auto-round").json()["run"]

    assert run["status"] == "converged"
    events = client.get(f"/runs/{run_id}/events").json()["events"]
    assert [event["event_type"] for event in events[:4]] == [
        "run_created",
        "initial_task",
        "local_submitted",
        "local_submitted",
    ]
    assert "x_star" in client.get(f"/runs/{run_id}").json()["run"]["reference"]


def test_http_manual_round_rejects_cross_round_and_timeout(isolated_app):
    client = TestClient(isolated_app.app)
    run_id = client.post("/runs", json=payload(max_iterations=5)).json()["run"]["run_id"]

    local_update(client, run_id, 1)
    task = client.get(f"/runs/{run_id}/task").json()["task"]
    item = next(w for w in task["workers"] if w["worker_id"] == 2)
    model = QuadraticModel(Q=np.asarray(item["Q"]), q=np.asarray(item["q"]))
    x2 = solve_local_update(model, z=np.asarray(item["z"]), u=np.asarray(item["u"]), rho=task["rho"], worker_id=2)
    stale = client.post(
        f"/runs/{run_id}/submit",
        json={"worker_id": 2, "round_number": 2, "rho": task["rho"], "x": x2.tolist()},
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "state_conflict"
    events = client.get(f"/runs/{run_id}/events").json()["events"]
    rejected = next(event for event in events if event["event_type"] == "submission_rejected")
    assert rejected["details"]["error_code"] == "state_conflict"

    timeout = client.post(f"/runs/{run_id}/timeout", json={"missing_worker_ids": [2]})
    assert timeout.status_code == 200
    assert timeout.json()["run"]["round_number"] == 1
    assert timeout.json()["run"]["pending_worker_ids"] == []

    local_update(client, run_id, 1)
    local_update(client, run_id, 2)
    done = client.post(f"/runs/{run_id}/complete")
    assert done.status_code == 200
    assert done.json()["run"]["round_number"] == 2


def test_http_rho_change_is_persisted(isolated_app):
    client = TestClient(isolated_app.app)
    run_id = client.post("/runs", json=payload(rho=0.05, adaptive_rounds=[1])).json()["run"]["run_id"]
    run = client.post(f"/runs/{run_id}/auto-round").json()["run"]
    assert np.isclose(run["rho"], 0.1)
    events = client.get(f"/runs/{run_id}/events").json()["events"]
    rho_event = next(event for event in events if event["event_type"] == "rho_changed")
    assert rho_event["details"]["old_rho"] == 0.05
    assert rho_event["details"]["new_rho"] == 0.1

    reloaded = TestClient(isolated_app.app).get(f"/runs/{run_id}").json()["run"]
    assert reloaded["rho"] == 0.1


def test_error_categories_for_bad_input_resource_and_missing_run(isolated_app):
    client = TestClient(isolated_app.app)
    bad = client.post(
        "/runs",
        json={
            "dimension": 2,
            "workers": [
                {"worker_id": 1, "Q": np.eye(2).tolist(), "q": [0, 0]},
                {"worker_id": 2, "Q": (-np.eye(2)).tolist(), "q": [0, 0]},
            ],
            "config": {"rho": 1},
        },
    )
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "input"

    too_large = client.post(
        "/runs",
        json={
            "dimension": 9,
            "workers": [
                {"worker_id": 1, "Q": np.eye(9).tolist(), "q": np.zeros(9).tolist()},
                {"worker_id": 2, "Q": np.eye(9).tolist(), "q": np.zeros(9).tolist()},
            ],
        },
    )
    assert too_large.status_code == 429
    assert too_large.json()["error"]["code"] == "resource_exhausted"

    missing = client.get("/runs/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


def test_two_worker_subprocesses_and_coordinator_reach_convergence(isolated_app):
    settings = isolated_app.settings
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "ADMM_DATA_DIR": str(settings.data_dir),
            "ADMM_MAX_DIMENSION": str(settings.max_dimension),
            "ADMM_MAX_ITERATIONS": str(settings.max_iterations),
            "ADMM_MAX_RUNS": str(settings.max_runs),
        }
    )
    import uvicorn

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    config = uvicorn.Config(isolated_app.app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    import threading

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.4)

    import httpx

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 5
    with httpx.Client(base_url=base_url, timeout=5.0) as api:
        while time.time() < deadline:
            try:
                if api.get("/health").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.02)
        else:
            raise AssertionError("uvicorn test server did not become ready")
        response = api.post("/runs", json=payload(max_iterations=80))
        assert response.status_code == 201, response.text
        run_id = response.json()["run"]["run_id"]
        processes = [
            subprocess.Popen(
                [sys.executable, "-m", "admm_consensus.worker", base_url, run_id, str(worker_id), "--poll-seconds", "0.01"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for worker_id in (1, 2)
        ]
        try:
            terminal = {"converged", "max_iterations", "failed"}
            while True:
                run = api.get(f"/runs/{run_id}").json()["run"]
                if len(run["pending_worker_ids"]) == 2:
                    api.post(f"/runs/{run_id}/complete").raise_for_status()
                if run["status"] in terminal:
                    break
                time.sleep(0.01)
            assert run["status"] == "converged"
        finally:
            for process in processes:
                process.terminate()
            outputs = [process.communicate(timeout=5) for process in processes]
            server.should_exit = True
            thread.join(timeout=5)
        assert all("submitted round 1" in stdout for stdout, _ in outputs)
