from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from lasso_backend.app import app


client = TestClient(app)


def _payload(known_problem, **overrides):
    payload = {
        "x": known_problem["x"].tolist(),
        "y": known_problem["y"].tolist(),
        "lambdas": known_problem["lambdas"].tolist(),
        "max_iter": 20_000,
        "tol": 1e-8,
    }
    payload.update(overrides)
    return payload


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_path_endpoint_returns_results_and_redacted_diagnostics(known_problem):
    response = client.post("/api/v1/lasso/path", json=_payload(known_problem))
    assert response.status_code == 200
    body = response.json()

    assert body["accepted"] is True
    assert body["decision"] == "accepted"
    assert len(body["points"]) == 4
    assert body["points"][0]["support"] == [0]
    assert body["points"][-1]["support"] == [0, 2, 4]

    diagnostics = body["diagnostics"]
    assert diagnostics["request_id"] == body["request_id"]
    assert diagnostics["shape"] == [140, 5]
    assert diagnostics["sparse"] is False
    assert "values" not in diagnostics
    assert all("x" not in event["state"] for event in diagnostics["events"])
    assert diagnostics["per_lambda"][0]["decision"] == "accepted"


def test_sparse_coo_endpoint_matches_dense(known_problem):
    coo = np.nonzero(known_problem["x"])
    sparse_payload = _payload(
        known_problem,
        x={
            "format": "coo",
            "n_rows": 140,
            "n_cols": 5,
            "rows": coo[0].tolist(),
            "cols": coo[1].tolist(),
            "data": known_problem["x"][coo].tolist(),
        },
    )

    sparse_response = client.post("/api/v1/lasso/path", json=sparse_payload)
    dense_response = client.post(
        "/api/v1/lasso/path", json=_payload(known_problem)
    )

    assert sparse_response.status_code == 200
    assert sparse_response.json()["diagnostics"]["sparse"] is True
    np.testing.assert_allclose(
        [p["coefficients"] for p in sparse_response.json()["points"]],
        [p["coefficients"] for p in dense_response.json()["points"]],
        atol=1e-8,
    )


def test_nonconvergence_is_http_200_undetermined(known_problem):
    response = client.post(
        "/api/v1/lasso/path",
        json=_payload(
            known_problem, lambdas=[0.001], max_iter=1, tol=1e-12
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is False
    assert body["decision"] == "undetermined"
    assert body["points"][0]["status"] == "max_iterations"
    assert body["points"][0]["converged"] is False
    assert body["diagnostics"]["per_lambda"][0]["decision"] == "undetermined"


def test_constant_column_rejection_has_specific_category_and_request_id():
    payload = {
        "x": [[0.0, 3.0], [1.0, 3.0], [2.0, 3.0], [3.0, 3.0]],
        "y": [1.0, 2.0, 3.0, 4.0],
        "lambdas": [0.1],
    }
    response = client.post("/api/v1/lasso/path", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["decision"] == "rejected"
    assert body["error_category"] == "invalid_preprocessing"
    assert "columns rejected" in body["message"]
    assert isinstance(body["request_id"], str)
    assert len(body["request_id"]) == 36


def test_shape_mismatch_is_rejected_before_solving(known_problem):
    response = client.post(
        "/api/v1/lasso/path",
        json=_payload(known_problem, y=known_problem["y"][:-1].tolist()),
    )

    assert response.status_code == 422
    body = response.json()
    assert body["decision"] == "rejected"
    assert body["error_category"] == "invalid_request"
    assert "same row count" in body["message"]


def test_duplicate_sparse_coordinates_are_rejected():
    payload = {
        "x": {
            "format": "coo",
            "n_rows": 3,
            "n_cols": 2,
            "rows": [0, 0],
            "cols": [1, 1],
            "data": [1.0, 2.0],
        },
        "y": [1.0, 2.0, 3.0],
        "lambdas": [0.1],
    }
    response = client.post("/api/v1/lasso/path", json=payload)

    assert response.status_code == 422
    assert "duplicate matrix coordinates" in response.text
