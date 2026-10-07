from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings


@pytest.fixture()
def client(tmp_path):
    settings = Settings(database_path=str(tmp_path / "test.db"), log_level="ERROR")
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def train_payload():
    return {
        "score_name": "risk_score",
        "target_name": "default",
        "extrapolation": "clip",
        "records": [
            {"score": 1.0, "target": 0},
            {"score": 2.0, "target": 1},
            {"score": 3.0, "target": 0},
            {"score": 4.0, "target": 1, "weight": 2.0},
            {"score": 5.0, "target": 1, "weight": 0.0},
        ],
    }


def test_train_predict_log_request_id(client):
    response = client.post("/v1/models", json=train_payload(), headers={"X-Request-ID": "req-123"})

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-123"
    assert body["status"] == "succeeded"
    assert body["result"]["score_name"] == "risk_score"
    assert body["result"]["zero_weight_count"] == 1
    assert body["warnings"][0]["code"] == "zero_weight"
    assert [step["module"] for step in body["steps"]] == [
        "validation",
        "kernel",
        "artifacts",
        "repository",
    ]
    probabilities = body["result"]["probabilities"]
    assert probabilities == sorted(probabilities)
    model_id = body["result"]["model_id"]

    prediction = client.post(
        "/v1/predictions",
        json={
            "score_name": "risk_score",
            "model_id": model_id,
            "records": [{"score": -5.0, "record_id": "x1"}, {"score": 4.0}],
        },
        headers={"X-Request-ID": "req-pred"},
    )
    assert prediction.status_code == 200
    prediction_body = prediction.json()
    assert prediction_body["result"]["predictions"][0]["extrapolated"] is True
    assert prediction_body["uncertainties"][0]["code"] == "extrapolated_probability"

    logs = client.get("/v1/request-logs/req-123").json()
    assert logs["result"]["logs"][0]["request_id"] == "req-123"
    assert logs["result"]["logs"][0]["status"] == "succeeded"
    assert "app.api" in logs["result"]["logs"][0]["processing_location"]


def test_unknown_model_returns_specific_failure(client):
    response = client.post(
        "/v1/predictions",
        json={"score_name": "risk", "model_id": "missing", "records": [{"score": 1.0}]},
    )

    assert response.status_code == 404
    body = response.json()
    assert body["status"] == "failed"
    assert body["failures"][0]["code"] == "model_not_found"


def test_evaluation_rejects_exact_calibration_set(client):
    trained = client.post("/v1/models", json=train_payload())
    model_id = trained.json()["result"]["model_id"]
    response = client.post(
        "/v1/evaluations",
        json={
            "score_name": "risk_score",
            "target_name": "default",
            "model_id": model_id,
            "records": train_payload()["records"],
        },
    )

    assert response.status_code == 422
    assert response.json()["failures"][0]["code"] == "evaluation_dataset_overlap"


def test_separate_evaluation_set_returns_metrics(client):
    trained = client.post("/v1/models", json=train_payload())
    model_id = trained.json()["result"]["model_id"]
    response = client.post(
        "/v1/evaluations",
        json={
            "score_name": "risk_score",
            "target_name": "default",
            "model_id": model_id,
            "records": [
                {"score": 1.5, "target": 0},
                {"score": 3.5, "target": 1},
                {"score": 4.2, "target": 1, "weight": 2.0},
            ],
        },
        headers={"X-Request-ID": "eval-1"},
    )

    assert response.status_code == 200
    metrics = response.json()["result"]["metrics"]
    assert metrics["count"] == 3
    assert metrics["total_weight"] == 4.0
    assert 0.0 <= metrics["weighted_brier_score"] <= 1.0
    assert metrics["weighted_log_loss"] > 0.0


def test_schema_failure_is_categorized(client):
    response = client.post(
        "/v1/models",
        json={"score_name": "risk", "target_name": "default", "records": [{"score": "bad", "target": 0}]},
    )

    assert response.status_code == 422
    assert response.json()["failures"][0]["code"] == "request_schema_invalid"
