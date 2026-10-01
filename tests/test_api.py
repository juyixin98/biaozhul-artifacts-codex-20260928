from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "requests.db"
    monkeypatch.setenv("SURVIVAL_SQLITE_PATH", str(db_path))

    import app.main as main_module
    import app.config as config_module

    monkeypatch.setattr(main_module, "settings", config_module.get_settings())
    monkeypatch.setattr(
        main_module,
        "processing_location",
        "test-host:test-version",
    )

    main_module.initialize_database(main_module.settings.sqlite_path)
    with TestClient(main_module.app) as client:
        yield client, main_module.settings.sqlite_path


def _hand_payload(request_id="req-test-001"):
    return {
        "request_id": request_id,
        "observations": [
            {"time": 2, "event": True, "group": "A"},
            {"time": 2, "event": False, "group": "A"},
            {"time": 3, "event": True, "group": "A"},
            {"time": 5, "event": False, "group": "A"},
            {"time": 6, "event": True, "group": "A"},
        ],
    }


def test_km_endpoint_returns_explainable_versioned_response(client):
    http, _ = client
    response = http.post("/api/v1/km", json=_hand_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-test-001"
    assert response.headers["x-request-id"] == "req-test-001"
    assert body["contract_version"] == "contract-v1"
    assert body["processing_location"] == "test-host:test-version"
    assert body["estimator_versions"] == ["km-greensill-loglog-v1"]
    assert [step["step"] for step in body["processing_steps"]] == [
        "parse-contract",
        "risk-set-order",
        "fit-kaplan-meier",
    ]
    timeline = body["result"]["groups"][0]["timeline"]
    assert [
        (p["time"], p["at_risk"], p["events"], p["censored"], p["survival"])
        for p in timeline
    ] == [
        (0, 5, 0, 0, 1.0),
        (2, 5, 1, 1, 0.8),
        (3, 3, 1, 0, 8 / 15),
        (5, 2, 0, 1, 8 / 15),
        (6, 1, 1, 0, 0.0),
    ]
    assert "Synthetic demonstration" in body["disclaimer"]


def test_logrank_endpoint_returns_specific_statistics(client):
    http, _ = client
    payload = {"observations": []}
    fixture = http.get("/api/v1/fixtures/logrank_hand").json()["observations"]
    payload["observations"] = fixture
    response = http.post("/api/v1/logrank", json=payload)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "valid"
    assert result["chi_square"] == pytest.approx(0.07390300230946893, abs=1e-14)
    assert result["p_value"] == pytest.approx(0.7857365379599127, abs=1e-14)


def test_contract_failure_has_specific_category_and_reasons(client):
    http, _ = client
    payload = _hand_payload()
    payload["observations"][0]["time"] = -1
    response = http.post("/api/v1/km", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["failure"]["code"] == "request_validation"
    reason = body["failure"]["reasons"][0]
    assert reason["category"] == "request_validation"
    assert "observations.0.time" in reason["location"]


def test_statistical_failure_separates_uncertain_from_failure(client):
    http, _ = client
    all_censored = {
        "observations": [
            {"time": 1, "event": False, "group": "A"},
            {"time": 2, "event": False, "group": "A"},
        ]
    }
    km = http.post("/api/v1/km", json=all_censored).json()
    median = km["result"]["groups"][0]["median"]
    assert median == {
        "reachable": False,
        "value": None,
        "interval_lower": None,
        "interval_upper": None,
        "status": "no_events",
        "reason": "No observed events exist, so the median event time is not estimable.",
    }

    logrank_payload = {
        "observations": [
            {"time": 1, "event": False, "group": "A"},
            {"time": 2, "event": False, "group": "B"},
        ]
    }
    logrank = http.post("/api/v1/logrank", json=logrank_payload).json()["result"]
    assert logrank["testable"] is False
    assert logrank["p_value"] is None
    assert logrank["notices"][0]["code"] == "no_events"


def test_sqlite_log_links_request_id_steps_and_status(client):
    http, db_path = client
    response = http.post(
        "/api/v1/km",
        json=_hand_payload("req-db-001"),
        headers={"X-Request-ID": "req-header-ignored"},
    )
    assert response.status_code == 200

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT request_id, path, status_code, processing_location,
                   estimator_versions, processing_steps
              FROM request_log
             WHERE request_id = ?
            """,
            ("req-db-001",),
        ).fetchone()

    assert row is not None
    assert row[1] == "/api/v1/km"
    assert row[2] == 200
    assert row[3] == "test-host:test-version"
    assert json.loads(row[4]) == ["km-greensill-loglog-v1"]
    steps = json.loads(row[5])
    assert steps[1]["detail"].startswith("Risk set is entry_time < t <= time")


def test_synthetic_dataset_is_deterministic_and_generated(client):
    http, _ = client
    payload = {"seed": 7, "sample_size_per_group": 3, "include_left_truncation": True}
    first = http.post("/api/v1/synthetic-dataset", json=payload).json()
    second = http.post("/api/v1/synthetic-dataset", json=payload).json()

    assert first == second
    assert first["metadata"]["seed"] == 7
    assert len(first["observations"]) == 6
    assert all(0 < row["entry_time"] < row["time"] for row in first["observations"])


def test_reproduce_endpoint_exposes_reference_results(client):
    http, _ = client
    body = http.post("/api/v1/reproduce").json()
    assert body["reference_library"] == {"name": "lifelines", "version": "0.30.3"}
    assert all(check["passed"] for check in body["checks"])
