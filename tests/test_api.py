from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from experiments.reference import scenarios
from lag_correlation.app import app, get_repository
from lag_correlation.repository import AnalysisRepository


@pytest.fixture()
def client(tmp_path) -> TestClient:
    repository = AnalysisRepository(str(tmp_path / "api.sqlite3"))
    app.dependency_overrides[get_repository] = lambda: repository
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def payload_for(scenario_name: str, request_id: str | None = None) -> dict:
    scenario = next(item for item in scenarios() if item.name == scenario_name)
    return {
        "request_id": request_id,
        "x": {
            "name": scenario.x_name,
            "times": scenario.x_times,
            "values": scenario.x_values,
        },
        "y": {
            "name": scenario.y_name,
            "times": scenario.y_times,
            "values": scenario.y_values,
        },
        "lag_grid": {
            "min_lag": scenario.lag_grid[0],
            "max_lag": scenario.lag_grid[1],
            "step": scenario.lag_grid[2],
        },
        "matching": {"tolerance": scenario.tolerance},
    }


def test_health_reports_versions(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["schema_version"] == "v1"


def test_analyze_returns_versioned_result_and_pair_details(client: TestClient) -> None:
    response = client.post(
        "/api/v1/lag-correlation/analyses",
        json=payload_for("known_offset_different_density", "api-success"),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["failure_category"] is None
    assert body["peak_lag"] == 2.0
    assert body["peak_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert body["schema_version"] == "v1"
    assert body["best_lag"]["n_pairs"] == 10
    assert body["best_lag"]["pair_details"][0]["paired_indices"] == [0, 1, 2]
    assert body["diagnostics"][0]["request_id"] == "api-success"
    assert "does not establish causality" in body["causality_notice"]

    fetched = client.get("/api/v1/lag-correlation/analyses/api-success")
    assert fetched.status_code == 200
    assert fetched.json()["request_id"] == "api-success"


def test_unknown_statistical_result_is_http_success_with_failure_category(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/lag-correlation/analyses",
        json=payload_for("constant_after_detrend", "api-constant"),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unknown"
    assert body["failure_category"] == "INSUFFICIENT_VARIANCE"
    assert body["peak_lag"] is None
    assert body["peak_correlation"] is None
    assert body["best_lag"]["status"] == "INSUFFICIENT_VARIANCE"


def test_invalid_series_returns_specific_validation_category(client: TestClient) -> None:
    payload = payload_for("known_offset_different_density", "api-invalid")
    payload["x"]["times"][1] = payload["x"]["times"][0]

    response = client.post(
        "/api/v1/lag-correlation/analyses",
        json=payload,
    )

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "error"
    assert body["failure_category"] == "VALIDATION_ERROR"
    assert "strictly increasing" in str(body["errors"])


def test_duplicate_request_id_is_conflict(client: TestClient) -> None:
    first = client.post(
        "/api/v1/lag-correlation/analyses",
        json=payload_for("uniform_sampling_special_case", "api-duplicate"),
    )
    second = client.post(
        "/api/v1/lag-correlation/analyses",
        json=payload_for("uniform_sampling_special_case", "api-duplicate"),
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["failure_category"] == "DUPLICATE_REQUEST_ID"


def test_missing_result_is_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/lag-correlation/analyses/not-present")

    assert response.status_code == 404
    assert response.json()["failure_category"] == "RESULT_NOT_FOUND"
