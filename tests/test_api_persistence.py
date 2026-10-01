from __future__ import annotations

import json
from pathlib import Path

from survey_backend.synthesis.generator import generate_survey


FIXTURE_PATH = Path("tests/fixtures/minimal_survey.json")


def _fixture_payload():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_returns_three_methods_diagnostics_and_masked_ids(client):
    payload = _fixture_payload()
    response = client.post(
        "/api/v1/synthetic/analyses",
        json=payload,
        headers={"X-Request-ID": "req-test-001", "X-Analysis-ID": "grp-test-001"},
    )
    assert response.status_code == 201
    body = response.json()
    assert [item["method"] for item in body["results"]] == [
        "complete_case",
        "mean_imputation",
        "multiple_imputation",
    ]
    assert body["version"] == 1
    assert body["request_id"] == "req-test-001"

    mi = body["results"][2]
    assert mi["runs"]
    assert len({run["stream_tag"] for run in mi["runs"]}) == body["imputations"]
    assert mi["pooling"][0]["within_variance"] > 0
    assert mi["pooling"][0]["total_variance"] >= mi["pooling"][0]["between_variance"]

    diagnostics = [diagnostic for item in body["results"] for diagnostic in item["diagnostics"]]
    assert any(diag["code"] == "eligible_scope" for diag in diagnostics)
    assert any(diag["code"] == "missing_distinguished_from_zero" for diag in diagnostics)
    assert any(diag["code"] == "missingness_mechanism_unobserved" for diag in diagnostics)
    assert all(diag["redacted_request_id"].startswith("re***") for diag in diagnostics)
    assert any("MNAR" in assumption for assumption in body["assumptions"])


def test_versioning_appends_and_identical_payload_is_idempotent(client):
    survey = generate_survey(n=50, seed=303, mechanism="mar", missing_probability=0.1)
    first_payload = {
        "dataset_id": "versioned-dataset",
        "request_id": "version-request-1",
        "seed": 303,
        "imputations": 2,
        "burn_in": 1,
        "records": [record.model_dump() for record in survey.to_records()],
    }
    first = client.post(
        "/api/v1/synthetic/analyses",
        json=first_payload,
        headers={"X-Analysis-ID": "version-group"},
    )
    assert first.status_code == 201
    assert first.json()["version"] == 1

    repeated = client.post(
        "/api/v1/synthetic/analyses",
        json=first_payload,
        headers={"X-Analysis-ID": "version-group"},
    )
    assert repeated.status_code == 201
    assert repeated.json()["version"] == 1
    assert repeated.json()["analysis_id"] == first.json()["analysis_id"]

    second_payload = dict(first_payload)
    second_payload["request_id"] = "version-request-2"
    second = client.post(
        "/api/v1/synthetic/analyses",
        json=second_payload,
        headers={"X-Analysis-ID": "version-group"},
    )
    assert second.status_code == 201
    assert second.json()["version"] == 2

    versions = client.get("/api/v1/synthetic/analyses/version-group/versions")
    assert versions.status_code == 200
    assert [row["version"] for row in versions.json()["versions"]] == [1, 2]


def test_all_missing_field_returns_specific_failure_class(client):
    survey = generate_survey(n=50, seed=202, make_q1_all_missing=True)
    payload = {
        "dataset_id": "bad-dataset",
        "records": [record.model_dump() for record in survey.to_records()],
        "seed": 202,
        "imputations": 2,
        "burn_in": 1,
    }
    response = client.post(
        "/api/v1/synthetic/analyses",
        json=payload,
        headers={"X-Request-ID": "bad-request"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "all_missing_field"
    assert body["details"]["failure_class"] == "AllMissingFieldError"
    assert body["redacted_request_id"].startswith("ba***")


def test_contract_rejects_infinity_and_marks_field(client):
    payload = _fixture_payload()
    payload["records"][0]["q1_screen_hours"] = float("inf")
    response = client.post("/api/v1/synthetic/analyses", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "invalid_contract"
    assert body["details"]["error_type"] == "value_error"


def test_missing_analysis_returns_404(client):
    response = client.get("/api/v1/synthetic/analyses/missing/versions/latest")
    assert response.status_code == 404
