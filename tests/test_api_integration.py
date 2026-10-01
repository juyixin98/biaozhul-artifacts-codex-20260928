"""End-to-end FastAPI and SQLite integration tests."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.api import app
from app.config import Settings
from app.repository import RunRepository
from app.service import CalibrationService


def make_payload(run_id="api-run"):
    return {
        "run_id": run_id,
        "fit": [
            {"id": "fit-0", "y": 0, "p": 0.1, "w": 1},
            {"id": "fit-1", "y": 0, "p": 0.2, "w": 1},
            {"id": "fit-2", "y": 1, "p": 0.8, "w": 1},
            {"id": "fit-3", "y": 1, "p": 0.9, "w": 1},
        ],
        "eval": [
            {"id": "eval-0", "y": 0, "p": 0.0, "w": 1},
            {"id": "eval-1", "y": 0, "p": 0.4, "w": 2},
            {"id": "eval-2", "y": 1, "p": 0.6, "w": 2},
            {"id": "eval-3", "y": 1, "p": 1.0, "w": 1},
        ],
        "calibrator": {"type": "isotonic"},
        "fixed_bins": {"edges": [0.0, 0.5, 1.0]},
        "quantile_bins": {"n_bins": 2},
        "metadata": {"test": True},
    }


def test_create_persist_and_fetch_run(tmp_path, run_identity, monkeypatch):
    db_path = tmp_path / "api.sqlite3"
    log_path = tmp_path / "api-test.log"
    service = CalibrationService(
        Settings(database_path=db_path, log_file=log_path), RunRepository(db_path)
    )
    monkeypatch.setattr("app.api.service", service)
    run_identity.emit("api_create_fetch", run_id="api-run")

    with TestClient(app) as client:
        response = client.post("/runs", json=make_payload())
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["status"] == "SUCCEEDED"
        assert body["run_id"] == "api-run"
        assert body["versions"]["kernel_version"]
        assert body["calibrator"]["disjoint_fit_and_evaluation"] is True
        assert body["analyses"]["raw"]["fixed"]["weight_conservation_check"]["passed"] is True

        fetched = client.get("/runs/api-run")
        assert fetched.status_code == 200
        assert fetched.json()["input_hash"] == body["input_hash"]

        listing = client.get("/runs")
        assert listing.status_code == 200
        assert [item["run_id"] for item in listing.json()] == ["api-run"]


def test_api_rejects_probability_out_of_bounds_without_claiming_success(run_identity):
    payload = make_payload("bad-probability")
    payload["eval"][0]["p"] = 1.5
    run_identity.emit("api_invalid_probability", failure_status="FAILED")

    with TestClient(app) as client:
        response = client.post("/runs", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "FAILED"
    assert body["category"] == "VALIDATION_ERROR"


def test_api_returns_404_for_unknown_run():
    with TestClient(app) as client:
        response = client.get("/runs/does-not-exist")
    assert response.status_code == 404
    assert response.json()["status"] == "FAILED"


def test_sample_payload_is_valid_contract():
    from pathlib import Path

    payload_path = Path(__file__).resolve().parents[1] / "sample_data" / "request.json"
    payload = json.loads(payload_path.read_text())
    from app.contracts import RunRequest

    request = RunRequest.model_validate(payload)
    assert request.calibrator.type == "isotonic"
    assert len(request.fit) == 4
    assert len(request.evaluation) == 4
