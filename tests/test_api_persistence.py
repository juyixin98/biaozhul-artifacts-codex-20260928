import sqlite3

from fastapi.testclient import TestClient

from changepoint_api.app import get_settings
from changepoint_api.config import Settings
from changepoint_api.app import create_app


def make_client(tmp_path):
    settings = Settings(
        db_path=tmp_path / "test.db",
        exact_dp_max_samples=30,
        pruned_dp_max_samples=20,
    )
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app), settings


def test_detection_response_contract_and_event_log_identity(tmp_path):
    client, settings = make_client(tmp_path)
    payload = {
        "samples": [0.0, 0.0, 5.0, 5.0],
        "penalty": 0.2,
        "min_segment_length": 2,
        "algorithm": "exact_dp",
    }

    response = client.post("/api/v1/detections", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "success"
    assert body["change_points"] == [2]
    assert body["num_segments"] == 2
    assert body["total_sse"] == 0.0
    assert body["objective"] == 0.4
    assert [segment["cost"] for segment in body["segments"]] == [0.0, 0.0]
    assert body["numeric_guard"]["prefix_strategy"].startswith(
        "median-anchored"
    )

    run_id = body["run_id"]
    status = client.get(f"/api/v1/runs/{run_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "success"
    assert status.json()["input_digest"] == body["input_digest"]

    events = client.get(f"/api/v1/runs/{run_id}/events").json()
    assert events
    assert all(event["details"]["run_id"] == run_id for event in events)
    assert all(
        event["details"]["input_digest"] == body["input_digest"]
        for event in events
    )
    assert {event["event_type"] for event in events} >= {
        "kernel_start",
        "dp_progress",
        "run_success",
    }
    progress = next(event for event in events if event["event_type"] == "dp_progress")
    assert progress["details"]["decision_basis"].startswith("minimize dp")
    assert progress["details"]["package_version"] == body["package_version"]

    with sqlite3.connect(settings.db_path) as connection:
        row = connection.execute(
            "SELECT status, error_category FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        segment_rows = connection.execute(
            "SELECT start_index, end_index, cost FROM run_segments "
            "WHERE run_id=? ORDER BY seq",
            (run_id,),
        ).fetchall()
    assert row == ("success", None)
    assert segment_rows == [(0, 2, 0.0), (2, 4, 0.0)]


def test_kernel_unsupported_configuration_is_persisted_as_failed(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post(
        "/api/v1/detections",
        json={
            "samples": [1.0, 2.0],
            "penalty": 1.0,
            "min_segment_length": 3,
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "unsupported_configuration"
    assert body["run_id"]
    assert "no legal segmentation" in body["message"]

    run = client.get(f"/api/v1/runs/{body['run_id']}").json()
    assert run["status"] == "failed"
    assert run["error"]["category"] == "unsupported_configuration"
    assert run["result"] is None


def test_invalid_sample_contract_returns_request_error_category(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post(
        "/api/v1/detections",
        content='{"samples":[1.0,"NaN"],"penalty":1.0}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert "finite" in response.text


def test_unknown_run_is_not_reported_as_success(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.get("/api/v1/runs/missing")

    assert response.status_code == 404
    assert response.json()["error_code"] == "run_not_found"


def test_sample_limit_failure_has_explicit_category(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post(
        "/api/v1/detections",
        json={
            "samples": [1.0] * 31,
            "penalty": 1.0,
            "algorithm": "exact_dp",
        },
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == "unsupported_configuration"
    assert "at most 30" in response.json()["message"]
