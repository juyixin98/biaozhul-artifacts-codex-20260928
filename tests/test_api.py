"""End-to-end API tests with concrete expected values and failure classes."""
import pytest
from fastapi.testclient import TestClient

from calibration_service.app.main import app

client = TestClient(app)

FIT_BODY = {
    "model_id": "api-model",
    "dataset": {
        "scores": [0.1, 0.2, 0.3, 0.4, 0.5],
        "labels": [0, 0, 1, 0, 1],
        "weights": [1, 1, 1, 1, 1],
    },
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_fit_returns_mapping_and_validation_report():
    r = client.post("/fit", json=FIT_BODY)
    assert r.status_code == 200
    body = r.json()
    assert body["request_id"]
    assert body["monotone_non_decreasing"] is True
    assert body["score_field"] == "model_score"
    assert body["extrapolation"] == "clamp_to_boundary"
    # hand-computed: labels at 0.3/0.4 pool to 0.5
    assert body["mapping"]["probabilities"] == [0.0, 0.0, 0.5, 1.0]
    assert body["mapping"]["thresholds"] == [0.1, 0.2, 0.4, 0.5]
    assert body["validation"]["n_used"] == 5


def test_predict_uses_fitted_step_function_and_clamps():
    client.post("/fit", json=FIT_BODY)
    r = client.post(
        "/predict",
        json={"model_id": "api-model", "scores": [-1.0, 0.35, 0.45, 7.0]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["probabilities"] == [0.0, 0.5, 1.0, 1.0]
    assert body["score_field"] == "model_score"


def test_evaluate_on_separate_set():
    client.post("/fit", json=FIT_BODY)
    r = client.post(
        "/evaluate",
        json={
            "model_id": "api-model",
            "dataset": {
                "scores": [0.15, 0.35, 0.45],
                "labels": [0, 1, 0],
            },
        },
    )
    assert r.status_code == 200
    m = r.json()["metrics"]
    assert m["n_eval_rows"] == 3
    # predictions: 0.0, 0.5, 1.0 -> brier = (0 + 0.25 + 1)/3
    assert m["brier_score"] == pytest.approx(1.25 / 3, abs=1e-12)
    assert m["log_loss"] > 0
    assert 0.0 <= m["ece"] <= 1.0


def test_fit_negative_weight_returns_categorized_error():
    r = client.post(
        "/fit",
        json={
            "model_id": "bad2",
            "dataset": {
                "scores": [0.1, 0.2],
                "labels": [0, 1],
                "weights": [1, -2],
            },
        },
    )
    assert r.status_code == 422
    body = r.json()
    assert body["category"] == "validation_error"
    assert body["detail"] == [1]
    assert body["request_id"]


def test_fit_bad_label_returns_validation_error():
    r = client.post(
        "/fit",
        json={
            "model_id": "bad3",
            "dataset": {"scores": [0.1, 0.2], "labels": [0, 2]},
        },
    )
    assert r.status_code == 422
    assert r.json()["category"] == "validation_error"


def test_predict_unknown_model_is_not_found():
    r = client.post(
        "/predict", json={"model_id": "does-not-exist", "scores": [0.1]}
    )
    assert r.status_code == 404
    assert r.json()["category"] == "not_found"


def test_request_id_header_is_present():
    r = client.get("/health")
    assert r.headers.get("X-Request-Id")
