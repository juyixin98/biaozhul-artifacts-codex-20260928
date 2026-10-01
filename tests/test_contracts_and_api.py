from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import create_app
from app.mcmc_diagnostics.models import MultiChainRequest, SingleChainRequest


def test_contract_rejects_non_finite_values():
    with pytest.raises(ValidationError) as exc_info:
        SingleChainRequest(values=[1.0, float("nan"), 3.0])
    assert "finite" in str(exc_info.value)


def test_contract_rejects_empty_and_duplicate_chain_ids():
    with pytest.raises(ValidationError, match="at least one"):
        MultiChainRequest(chains=[])
    with pytest.raises(ValidationError, match="unique"):
        MultiChainRequest(chains=[[1.0, 2.0], [3.0, 4.0]], chain_ids=["a", "a"])


def test_api_single_chain_persists_and_can_fetch(tmp_path):
    from app.mcmc_diagnostics.dependencies import cached_settings

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/diagnostics/single",
            json={"request_id": "single-1", "values": [float(x) for x in range(1, 81)]},
        )
        assert response.status_code == 200
        assert response.headers["X-Request-Id"] == "single-1"
        assert response.headers["X-Kernel-Version"] == cached_settings().kernel_version
        row_id = response.headers["X-SQLite-Row-Id"]

        fetched = client.get("/api/v1/runs/single-1")
        assert fetched.status_code == 200
        assert fetched.json()["id"] == int(row_id)
        assert fetched.json()["request_id"] == "single-1"
        assert json.loads(fetched.json()["result_json"])["status"] in {
            "ok",
            "warning",
            "inconclusive",
        }


def test_api_multi_chain_returns_structured_failure_for_inf():
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/diagnostics/multi",
            content='{"request_id":"bad-1","chains":[[1.0,NaN],[1.0,2.0]]}',
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
        body = response.json()
        assert body["request_id"] == "bad-1"
        assert body["error"]["code"] == "SAMPLE_CONTRACT_VIOLATION"
        assert body["uncertainties"]
        assert response.headers["X-Request-Id"] == "bad-1"

        fetched = client.get("/api/v1/runs/bad-1")
        assert fetched.status_code == 200
        assert fetched.json()["error_code"] == "SAMPLE_CONTRACT_VIOLATION"


def test_api_unequal_length_error_is_explicit():
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/diagnostics/multi",
            json={"chains": [[1.0, 2.0, 3.0], [1.0, 2.0]]},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_SAMPLE_SHAPE"


def test_result_explains_steps_versions_and_limits():
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/diagnostics/multi",
            json={"chains": [[1.0, 2.0, 3.0, 4.0] * 10] * 2},
        )
        body = response.json()
        assert body["versions"]["kernel_version"]
        assert body["versions"]["algorithm_revision"]
        assert len(body["steps"]) >= 4
        assert any("do not prove convergence" in item for item in body["limitations"])
        assert any("within-chain variance" in item for item in body["steps"])
