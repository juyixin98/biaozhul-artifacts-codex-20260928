from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from change_detection.api import create_app, get_service, get_settings
from change_detection.config import Settings
from change_detection.errors import PruningVerificationError
from change_detection.service import ChangePointService


@pytest.fixture
def client(tmp_path) -> TestClient:
    settings = Settings(
        database_path=str(tmp_path / "api.db"),
        max_series_length=1000,
        log_level="DEBUG",
        log_path=str(tmp_path / "api.log"),
    )
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_service] = lambda: ChangePointService(settings)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_health_reports_database_and_versions(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] is True
    assert body["core_version"] == "1.0.0"
    assert body["schema_version"] == 1


def test_analyze_returns_segments_cost_total_and_persisted_run(client: TestClient) -> None:
    response = client.post(
        "/api/v1/change-points/analyze",
        json={
            "values": [0, 0, 0, 10, 10, 10],
            "min_segment_length": 1,
            "penalty": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCEEDED"
    assert body["objective"] == 0.0
    assert body["sse"] == 0.0
    assert body["total_penalty"] == 0.0
    assert body["breakpoints"] == [3]
    assert body["verification_mode"] == "matched_unpruned_dp"
    assert [segment["cost"] for segment in body["segments"]] == [0.0, 0.0]

    stored = client.get(f"/api/v1/runs/{body['run_id']}")
    assert stored.status_code == 200
    assert stored.json()["status"] == "SUCCEEDED"
    assert stored.json()["input_sha256"] == body["input_sha256"]
    assert any(event["payload"]["run_id"] == body["run_id"] for event in stored.json()["events"])


def test_api_rejects_non_finite_input_with_422_instead_of_success(client: TestClient) -> None:
    response = client.post(
        "/api/v1/change-points/analyze",
        json={"values": [1.0, float("nan")], "penalty": 0},
    )

    assert response.status_code == 422


def test_api_returns_named_422_for_min_segment_too_large(client: TestClient) -> None:
    response = client.post(
        "/api/v1/change-points/analyze",
        json={"values": [1, 2], "min_segment_length": 3, "penalty": 0},
    )

    assert response.status_code == 422
    assert response.json()["category"] == "INVALID_PARAMETER"


def test_unknown_domain_failure_is_500_not_success(monkeypatch, client: TestClient) -> None:
    def boom(*args, **kwargs):
        raise PruningVerificationError("forced mismatch", details={"reason": "test"})

    monkeypatch.setattr("change_detection.service.solve", boom)
    response = client.post(
        "/api/v1/change-points/analyze",
        json={"values": [1, 2], "penalty": 0},
    )

    assert response.status_code == 500
    body = response.json()
    assert body["category"] == "PRUNING_MISMATCH"
    assert body["details"] == {"reason": "test"}


def test_missing_run_is_404(client: TestClient) -> None:
    response = client.get("/api/v1/runs/no-such-run")

    assert response.status_code == 404
    assert response.json()["category"] == "RUN_NOT_FOUND"
