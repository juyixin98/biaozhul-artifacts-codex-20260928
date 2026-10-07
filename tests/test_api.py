"""API-level tests for the FastAPI report interface."""
import os
import tempfile
from pathlib import Path

import pytest

os.environ["MUTATION_DB_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="mut-api-test-"), "api_test.db"
)

from fastapi.testclient import TestClient

from app.report import app

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

client = TestClient(app)


def test_health_reports_versions():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["operator_set_version"] == "ops-v1"
    assert "X-Request-ID" in resp.headers


def test_request_id_echoed():
    resp = client.get("/health", headers={"X-Request-ID": "req-123"})
    assert resp.headers["X-Request-ID"] == "req-123"


@pytest.fixture(scope="module")
def completed_run():
    resp = client.post("/runs/sync", json={
        "source_dir": str(FIXTURES / "startup_pkg"),
        "test_dir": "tests",
        "timeout_seconds": 15,
        "requester": "api-test",
    })
    assert resp.status_code == 200
    return resp.json()


def test_sync_run_and_fetch(completed_run):
    assert completed_run["status"] == "completed"
    assert completed_run["requester"] == "api-test"
    assert completed_run["startup_failure"] == 1
    run_id = completed_run["run_id"]

    fetched = client.get(f"/runs/{run_id}")
    assert fetched.status_code == 200
    assert fetched.json()["run_id"] == run_id

    mutants = client.get(f"/runs/{run_id}/mutants").json()
    assert len(mutants) == 1
    assert mutants[0]["outcome"] == "startup_failure"
    assert mutants[0]["file_path"] == "config_mod.py"


def test_report_separates_failures_and_undetermined(completed_run):
    run_id = completed_run["run_id"]
    report = client.get(f"/runs/{run_id}/report").json()
    assert report["summary"]["run_id"] == run_id
    assert len(report["failures"]) == 1
    assert report["failures"][0]["outcome"] == "startup_failure"
    assert report["undetermined"] == []
    assert report["survived_mutants"] == []


def test_unknown_run_404():
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/runs/nope/mutants").status_code == 404
    assert client.get("/runs/nope/report").status_code == 404


def test_run_request_validation():
    resp = client.post("/runs/sync", json={"source_dir": ""})
    assert resp.status_code == 422
    resp = client.post("/runs/sync", json={
        "source_dir": "/tmp/x", "timeout_seconds": -5,
    })
    assert resp.status_code == 422
