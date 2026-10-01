from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.errors import ComputationError
from app.service import BetaBinomialService
from app.storage import SQLiteStore


@pytest.fixture
def client() -> TestClient:
    service = BetaBinomialService(SQLiteStore(":memory:"))
    return TestClient(create_app(service))


def test_full_api_contract_and_interval_naming(client: TestClient) -> None:
    assert client.post(
        "/runs", json={"run_id": "r1", "alpha": 1, "beta": 1}
    ).status_code == 200
    submit = client.post(
        "/runs/r1/batches/A", json={"successes": 3, "failures": 1}
    )
    assert submit.status_code == 200
    assert submit.json()["event_type"] == "batch_submitted"

    revise = client.put(
        "/runs/r1/batches/A",
        json={"successes": 1, "failures": 1, "reason": "correction"},
    )
    assert revise.status_code == 200
    assert revise.json()["event_type"] == "batch_revised"

    posterior = client.get("/runs/r1/posterior").json()
    assert posterior["posterior_alpha"] == 2
    assert posterior["posterior_beta"] == 2
    assert "credible_interval" in posterior
    assert "confidence_interval" in posterior

    prediction = client.post(
        "/runs/r1/predictions",
        json={"future_n": 3, "confidence_level": 0.95},
    )
    assert prediction.status_code == 200
    body = prediction.json()
    assert len(body["points"]) == 4
    assert body["probability_sum"] == pytest.approx(1.0, abs=1e-15)

    versions = client.get("/runs/r1/versions").json()
    assert [item["event_type"] for item in versions] == [
        "created",
        "batch_submitted",
        "batch_revised",
    ]


@pytest.mark.parametrize(
    ("payload", "detail_substring"),
    [
        ({"run_id": "bad", "alpha": 0, "beta": 1}, "greater than"),
        ({"run_id": "bad", "alpha": -1, "beta": 1}, "greater than"),
        ({"run_id": "bad", "alpha": "1", "beta": 1}, "numeric"),
        ({"run_id": "bad", "alpha": True, "beta": 1}, "numeric"),
    ],
)
def test_invalid_prior_returns_contract_category(
    client: TestClient, payload: dict, detail_substring: str
) -> None:
    response = client.post("/runs", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["category"] == "invalid_input"
    assert detail_substring in str(body["details"]).lower()


def test_non_finite_prior_query_is_rejected_as_invalid_input(
    client: TestClient,
) -> None:
    # Python's JSON client deliberately refuses to emit non-standard NaN. The
    # parser-level Infinity case verifies the same finite-prior contract.
    response = client.post(
        "/runs",
        content=b'{"run_id":"bad","alpha":Infinity,"beta":1}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["category"] == "invalid_input"


def test_count_contracts_and_state_error_categories(client: TestClient) -> None:
    client.post("/runs", json={"run_id": "r", "alpha": 1, "beta": 1})
    client.post("/runs/r/batches/A", json={"successes": 1, "failures": 1})

    duplicate = client.post(
        "/runs/r/batches/A", json={"successes": 1, "failures": 1}
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["category"] == "state_conflict"

    missing_revision = client.put(
        "/runs/missing/batches/A", json={"successes": 1, "failures": 1}
    )
    assert missing_revision.status_code == 404
    assert missing_revision.json()["category"] == "state_conflict"

    negative = client.post(
        "/runs/r/batches/B", json={"successes": "1", "failures": 0}
    )
    assert negative.status_code == 422
    assert negative.json()["category"] == "invalid_input"

    huge = client.post(
        "/runs/r/predictions", json={"future_n": 10_001}
    )
    assert huge.status_code == 413
    assert huge.json()["category"] == "resource_exhausted"


def test_computation_failure_has_distinct_category(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_prediction(*args, **kwargs):
        raise ComputationError("forced numerical failure", {"kernel": "beta-binomial"})

    client.post("/runs", json={"run_id": "r", "alpha": 1, "beta": 1})
    monkeypatch.setattr(
        "app.service.beta_binomial_predictions", broken_prediction
    )
    response = client.post(
        "/runs/r/predictions", json={"future_n": 2}
    )
    assert response.status_code == 500
    body = response.json()
    assert body["category"] == "computation_failure"
    assert body["error"] == "forced numerical failure"
