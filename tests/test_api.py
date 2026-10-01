from __future__ import annotations

import json
import sqlite3

import numpy as np

from app.fixtures import build_fixtures


def test_health_and_request_id(client, temp_db) -> None:
    response = client.get("/health", headers={"x-request-id": "health-123"})

    assert response.status_code == 200
    assert response.json()["requestId"] == "health-123"
    assert response.headers["x-request-id"] == "health-123"


def test_estimate_returns_versions_steps_weights_and_persists(client, temp_db) -> None:
    fixture = build_fixtures()["strong_outliers"]
    payload = {
        "data": fixture.data.tolist(),
        "method": "robust_tyler_shrinkage",
        "shrinkage": 0.25,
        "tolerance": 1e-12,
        "maxIterations": 1000,
    }
    response = client.post(
        "/api/v1/covariance/estimate", json=payload, headers={"x-request-id": "req-42"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["requestId"] == "req-42"
    assert body["schemaVersion"] == "covariance-response/v1"
    assert body["estimatorVersion"] == "covariance-kernel/1.0.0"
    assert body["status"] == "ok"
    assert body["invertible"] is True
    assert body["rank"] == 5
    assert body["outlierWeights"][0] < 0.01
    assert body["assumptions"]
    assert [step["step"] for step in body["steps"]] == [
        "sample.prepare_fixed_contract",
        "tyler.initialize",
        "tyler.fixed_point",
        "matrix.validate_independent_checks",
        "covariance.restore_original_units",
        "response.complete",
    ]

    with sqlite3.connect(temp_db) as connection:
        row = connection.execute(
            "SELECT status, failures_json, uncertainties_json FROM requests WHERE request_id = ?",
            ("req-42",),
        ).fetchone()
        step_count = connection.execute(
            "SELECT COUNT(*) FROM request_steps WHERE request_id = ?", ("req-42",)
        ).fetchone()[0]
    assert row[0] == "ok"
    assert json.loads(row[1]) == []
    assert json.loads(row[2]) == []
    assert step_count == 6


def test_missing_data_has_uncertainty_and_fixed_imputation(client) -> None:
    fixture = build_fixtures()["missing_values"]
    response = client.post(
        "/api/v1/covariance/estimate",
        json={"data": np.where(np.isnan(fixture.data), None, fixture.data).tolist()},
        headers={"x-request-id": "missing-1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "uncertain"
    codes = {item["code"] for item in body["uncertainties"]}
    assert "MISSING_IMPUTED" in codes
    assert body["missingMask"][0][0] is True
    assert body["imputedValues"][0][0] is not None


def test_ragged_matrix_returns_specific_failure_category(client) -> None:
    response = client.post(
        "/api/v1/covariance/estimate",
        json={"data": [[1.0, 2.0], [3.0]]},
        headers={"x-request-id": "bad-1"},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "value_error"


def test_bad_scales_returns_named_failure_category(client) -> None:
    response = client.post(
        "/api/v1/covariance/estimate",
        json={"data": [[1.0, 2.0]], "columnScales": [1.0, -2.0]},
        headers={"x-request-id": "bad-scale"},
    )

    assert response.status_code == 422
    assert "BAD_SCALES" in response.text
