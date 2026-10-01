from fastapi.testclient import TestClient

from app.main import create_app
from core.config import Settings


def client_for(tmp_path):
    settings = Settings(
        database_path=str(tmp_path / "integration.db"),
        log_path=str(tmp_path / "integration.jsonl"),
    )
    return TestClient(create_app(settings)), settings


def test_health_and_version_report_versions(tmp_path):
    client, _ = client_for(tmp_path)

    assert client.get("/health").json()["status"] == "ok"
    version = client.get("/version").json()
    assert version["algorithm_version"] == "calibration-core-0.1.0"
    assert version["schema_version"] == 1


def test_analysis_endpoint_returns_concrete_bins_and_persists_get(tmp_path):
    client, _ = client_for(tmp_path)
    payload = {
        "y": [0, 0, 1, 1],
        "p": [0.2, 0.2, 0.8, 0.8],
        "w": [1, 1, 1, 1],
        "method": "fixed",
        "edges": [0.0, 0.5, 1.0],
        "run_label": "integration fixed",
    }

    response = client.post("/api/v1/analyses", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["occupied_bins"] == 2
    assert body["empty_bins"] == 0
    assert body["decomposition"]["raw_brier"] == 0.04000000000000001
    assert len(body["bins"]) == 2

    stored = client.get(f"/api/v1/runs/{body['run_id']}")
    assert stored.status_code == 200
    assert stored.json()["status"] == "success"
    assert len(stored.json()["bins"]) == 2


def test_out_of_range_probability_returns_400_failed_category_and_stored_run(tmp_path):
    client, _ = client_for(tmp_path)

    response = client.post("/api/v1/analyses", json={"y": [0], "p": [1.2], "method": "fixed", "n_bins": 2})

    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "failed"
    assert body["error_category"] == "probability_range_error"


def test_calibration_endpoint_separates_train_and_evaluation(tmp_path):
    client, _ = client_for(tmp_path)
    payload = {
        "train": {"y": [0, 1, 0, 1], "p": [0.2, 0.2, 0.8, 0.8]},
        "evaluate": {"y": [1, 0], "p": [0.2, 0.8]},
        "method": "quantile",
        "n_bins": 2,
    }

    response = client.post("/api/v1/calibrations", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["transformed_probabilities"] == [0.5, 0.5]
    assert body["evaluation_analysis"]["status"] == "success"
    stored = client.get(f"/api/v1/runs/{body['run_id']}")
    assert {row["stage"] for row in stored.json()["bins"]} == {"train", "evaluation"}


def test_empty_calibration_bin_is_a_failure_not_smoothed(tmp_path):
    client, _ = client_for(tmp_path)
    payload = {
        "train": {"y": [0, 1], "p": [0.2, 0.8]},
        "evaluate": {"y": [1], "p": [0.5]},
        "method": "fixed",
        "n_bins": 10,
    }

    response = client.post("/api/v1/calibrations", json=payload)

    assert response.status_code == 400
    assert response.json()["error_category"] == "empty_calibration_bin"


def test_missing_run_is_404_failed_not_silent_success(tmp_path):
    client, _ = client_for(tmp_path)

    response = client.get("/api/v1/runs/missing")

    assert response.status_code == 404
    assert response.json()["status"] == "failed"
    assert response.json()["error_category"] == "run_not_found"
