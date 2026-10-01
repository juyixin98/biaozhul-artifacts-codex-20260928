from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SURVEY_DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("SURVEY_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("SURVEY_REDACT", "1")

    # Import after environment configuration so module-level create_app() uses
    # the isolated test database.
    from survey_imputation.api import app

    with TestClient(app) as client:
        yield client


def _minimal_payload(method: str = "mean_imputation", **overrides):
    records = [
        {"record_id": "nonuser-001", "age_z": -1.0, "member": 0, "used_support": 0, "contact_frequency": 0, "rating": 0},
        {"record_id": "nonuser-002", "age_z": 0.0, "member": 0, "used_support": 0, "contact_frequency": 0, "rating": 0},
        {"record_id": "member-003", "age_z": 0.0, "member": 1, "used_support": 1, "contact_frequency": 3, "rating": 7},
        {"record_id": "member-004", "age_z": 1.0, "member": 1, "used_support": 1, "contact_frequency": 4, "rating": None},
        {"record_id": "member-005", "age_z": 0.0, "member": 1, "used_support": 1, "contact_frequency": 5, "rating": 8},
        {"record_id": "guest-006", "age_z": -1.0, "member": 0, "used_support": 1, "contact_frequency": 2, "rating": None},
        {"record_id": "guest-007", "age_z": 0.0, "member": 0, "used_support": 1, "contact_frequency": 3, "rating": 6},
        {"record_id": "member-008", "age_z": 1.0, "member": 1, "used_support": 1, "contact_frequency": 6, "rating": 9},
        {"record_id": "member-009", "age_z": 0.0, "member": 1, "used_support": 1, "contact_frequency": 4, "rating": None},
        {"record_id": "member-010", "age_z": 0.0, "member": 1, "used_support": 1, "contact_frequency": 3, "rating": 7},
    ]
    payload = {"request_id": "api-mean-case", "records": records, "method": method, "seed": 1729}
    payload.update(overrides)
    return payload


def test_health_and_versions(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    versions = client.get("/versions").json()
    assert versions["contract"] == "1.0.0"
    assert versions["core"].startswith("core-")


def test_generate_synthetic_dataset(client):
    response = client.post(
        "/datasets/generate",
        json={
            "request_id": "api-generate",
            "n": 200,
            "mechanism": "mar",
            "missing_probability": 0.2,
            "seed": 1729,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["n_missing"] == 42
    assert body["true_mean_rating"] == 5.368195000000001
    assert any(item["code"] == "zero_missing_separation" for item in body["diagnostics"])


def test_analysis_endpoint_persists_version_and_redacts_identifiers(client):
    response = client.post("/analyses", json=_minimal_payload())

    assert response.status_code == 201
    body = response.json()
    assert body["request_id"] == "api-mean-case"
    assert body["estimate"]["estimate"] == pytest.approx(5.285714285714286)
    assert body["estimate"]["variance"] == pytest.approx(0.926984126984127)
    assert body["estimate"]["model_version"] == "mean-1.0.0"
    assert any(d["status"] == "undetermined" and d["code"] == "mar_identifiability" for d in body["diagnostics"])

    stored = client.get("/analyses/api-mean-case")
    assert stored.status_code == 200
    assert stored.json()["estimate"]["estimate"] == body["estimate"]["estimate"]


def test_multiple_imputation_returns_independent_previews(client):
    response = client.post(
        "/analyses",
        json=_minimal_payload(method="multiple_imputation", request_id="api-mi-case", m=4),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["estimate"]["imputation_count"] == 4
    assert body["estimate"]["marked_imputed"] == 12
    assert len(body["imputations_preview"]) == 9
    record_ids = {item["record_id"] for item in body["imputations_preview"]}
    assert all("***" in record_id for record_id in record_ids)
    first_imputation = [item["rating"] for item in body["imputations_preview"] if item["imputation_index"] == 1]
    second_imputation = [item["rating"] for item in body["imputations_preview"] if item["imputation_index"] == 2]
    assert first_imputation != second_imputation


def test_all_missing_field_is_rejected_with_category_and_diagnostics(client):
    records = _minimal_payload()["records"]
    for record in records:
        record["rating"] = None

    response = client.post(
        "/analyses",
        json={"request_id": "all-missing-case", "records": records, "method": "mean_imputation"},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["category"] == "analysis_rejected"
    diagnostic_codes = {item["code"] for item in body["diagnostics"]}
    assert "all_outcome_missing" in diagnostic_codes
    assert "observed_training_rows" in diagnostic_codes
    rejected = [item for item in body["diagnostics"] if item["code"] == "all_outcome_missing"]
    assert rejected[0]["status"] == "rejected"


def test_complete_data_rejects_missing_ratings(client):
    response = client.post("/analyses", json=_minimal_payload(method="complete_data"))

    assert response.status_code == 422
    assert response.json()["error"]["category"] == "analysis_rejected"


def test_invalid_probability_is_request_validation_error(client):
    response = client.post(
        "/datasets/generate",
        json={"n": 200, "mechanism": "mcar", "missing_probability": 0.9, "seed": 1},
    )

    assert response.status_code == 422


def test_missing_result_returns_not_found_category(client):
    response = client.get("/analyses/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["category"] == "result_not_found"


def test_duplicate_request_id_is_rejected_with_conflict_category(client):
    payload = _minimal_payload(request_id="duplicate-id")
    first = client.post("/analyses", json=payload)
    second = client.post("/analyses", json=payload)

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["error"]["category"] == "result_already_exists"
