from __future__ import annotations

import pytest

from lagcorr.api import create_app
from lagcorr.config import Settings


def make_payload(request_id: str = "case-1", include_pairs: bool = True):
    return {
        "request_id": request_id,
        "a": {
            "times": [float(i) for i in range(12)],
            "values": [float(i) for i in range(12)],
        },
        "b": {
            "times": [i + 0.5 for i in range(12)],
            "values": [float(i) for i in range(12)],
        },
        "lags": [0.5, 10.0],
        "tolerance": 0.6,
        "include_pairs": include_pairs,
    }


@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient

    app = create_app(Settings(database_path=str(tmp_path / "test.db")))
    with TestClient(app) as test_client:
        yield test_client


def test_create_and_fetch_analysis_with_statuses_and_diagnostics(client):
    response = client.post("/analyses", json=make_payload())
    assert response.status_code == 201
    body = response.json()

    assert body["request_id"] == "case-1"
    assert body["result_version"] == "irregular-lag-pearson-1.0.0"
    assert body["peak_lag"] == 0.5
    assert body["peak_correlation"] == pytest.approx(1.0)
    assert body["status_summary"] == {"accepted": 1, "unknown": 1}

    accepted, unknown = body["results"]
    assert accepted["status"] == "accepted"
    assert accepted["correlation"] == pytest.approx(1.0)
    assert accepted["pair_count"] == 12
    assert len(accepted["pairs"]) == 12
    assert accepted["pairs"][0] == {
        "a_index": 0,
        "b_index": 0,
        "a_time": 0.0,
        "b_time": 0.5,
        "a_value": 0.0,
        "b_value": 0.0,
        "aligned_b_time": 0.0,
        "residual": 0.0,
    }
    assert accepted["diagnosis"]["candidate_edges"] == 12
    assert unknown["status"] == "unknown"
    assert unknown["reason_code"] == "INSUFFICIENT_PAIR_COUNT"

    fetched = client.get("/analyses/case-1")
    assert fetched.status_code == 200
    assert fetched.json()["results"][0]["pairs"][0]["a_index"] == 0


def test_pair_evidence_can_be_omitted_without_changing_statistic(client):
    response = client.post("/analyses", json=make_payload("case-no-pairs", False))
    assert response.status_code == 201
    result = response.json()["results"][0]
    assert result["pairs"] == []
    assert result["correlation"] == pytest.approx(1.0)


def test_duplicate_request_id_has_specific_failure_category(client):
    client.post("/analyses", json=make_payload())
    duplicate = client.post("/analyses", json=make_payload())
    assert duplicate.status_code == 409
    assert duplicate.json()["error_code"] == "DUPLICATE_REQUEST_ID"


def test_invalid_series_returns_specific_failure_category(client):
    payload = make_payload("bad")
    payload["a"]["times"].append(1.0)
    payload["a"]["values"].append(1.0)
    response = client.post("/analyses", json=payload)
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_SERIES"
    assert "unique" in response.json()["message"]


def test_unknown_identifier_is_not_an_accepted_result(client):
    response = client.get("/analyses/missing")
    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "ANALYSIS_NOT_FOUND"


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
