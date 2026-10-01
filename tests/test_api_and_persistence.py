from __future__ import annotations

import sqlite3

import numpy as np


def test_health_uses_configured_sqlite_database(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["database"].endswith("test-diagnostics.db")


def test_diagnostics_response_correlation_versions_steps_and_persistence(client):
    rng = np.random.default_rng(101)
    payload = {
        "request_id": "corr-001",
        "chains": rng.normal(size=(2, 100)).tolist(),
        "max_acf_lag": 5,
    }
    response = client.post("/api/v1/diagnostics", json=payload)
    assert response.status_code == 200
    envelope = response.json()
    assert envelope["request_id"] == "corr-001"
    assert envelope["schema_version"] == "mcmc-diagnostics/v1"
    assert envelope["service_version"] == "1.0.0"
    assert envelope["processing_location"] == "local-fastapi-sqlite"
    result = envelope["result"]
    assert [step["step"] for step in result["key_steps"]] == [
        "validate",
        "demean_and_normalize_acf",
        "separate_batch_variance",
        "ess_truncation",
        "convergence_diagnostic",
    ]
    assert result["chains"][0]["acf"]["requested_max_lag"] == 5

    stored = client.get("/api/v1/results/corr-001")
    assert stored.status_code == 200
    assert stored.json()["request_id"] == "corr-001"
    events = client.get("/api/v1/results/corr-001/events").json()["events"]
    assert [event["stage"] for event in events] == ["received", "stored"]
    assert all(event["request_id"] == "corr-001" for event in events)


def test_constant_request_is_200_but_has_separate_inconclusive_diagnosis(client):
    response = client.post(
        "/api/v1/diagnostics",
        json={"request_id": "constant-001", "chains": [[1.0] * 8, [1.0] * 8]},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "inconclusive"
    assert result["findings"][0]["code"] == "CONSTANT_CHAIN"
    assert result["uncertainty_conclusions"]


def test_unequal_chains_return_specific_failure_category(client):
    response = client.post(
        "/api/v1/diagnostics",
        json={"request_id": "bad-shape", "chains": [[1.0, 2.0], [1.0]]},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["request_id"] == "bad-shape"
    assert body["error_code"] == "EQUAL_LENGTH_REQUIRED"
    assert body["schema_version"] == "mcmc-diagnostics/v1"
    events = client.get("/api/v1/results/bad-shape/events").json()["events"]
    assert events[-1]["stage"] == "request_contract_failed"
    assert events[-1]["details"]["error_code"] == "EQUAL_LENGTH_REQUIRED"


def test_empty_chain_returns_dedicated_failure_category(client):
    response = client.post(
        "/api/v1/diagnostics",
        json={"request_id": "empty-chain", "chains": [[1.0], []]},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "EMPTY_CHAIN"


def test_header_request_id_is_used_when_body_omits_it(client):
    response = client.post(
        "/api/v1/diagnostics",
        headers={"X-Request-ID": "header-correlation"},
        json={"chains": [[0.1, 0.2, 0.3, 0.4]], "persist": False},
    )
    assert response.status_code == 200
    assert response.json()["request_id"] == "header-correlation"
    assert client.get("/api/v1/results/header-correlation").status_code == 404
    events = client.get("/api/v1/results/header-correlation/events")
    assert events.status_code == 200
    assert events.json()["events"][0]["stage"] == "received"
    assert events.json()["events"][1]["stage"] == "completed_unpersisted"


def test_stored_json_is_valid_sqlite_text_without_non_finite_tokens(client):
    payload = {"request_id": "offset-constant", "chains": [[1.0] * 6, [5.0] * 6]}
    response = client.post("/api/v1/diagnostics", json=payload)
    assert response.status_code == 200

    database = client.get("/healthz").json()["database"]
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT result_json FROM analysis_runs WHERE request_id = ?",
            ("offset-constant",),
        ).fetchone()
    assert row is not None
    assert "NaN" not in row[0]
    assert "Infinity" not in row[0]
    assert "plus_infinity" in row[0]
