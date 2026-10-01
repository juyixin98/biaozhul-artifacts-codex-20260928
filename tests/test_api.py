from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app


client = TestClient(create_app())


def make_payload(dataset_id=None):
    payload = {
        "objectives": [
            {"name": "cost", "direction": "min"},
            {"name": "reliability", "direction": "max"},
        ],
        "candidates": [
            {"id": "A", "values": {"cost": 1, "reliability": 3}},
            {"id": "D", "values": {"cost": 1, "reliability": 3}},
            {"id": "E", "values": {"cost": 2, "reliability": 3}},
            {"id": "C", "values": {"cost": 2, "reliability": 2}},
            {"id": "B", "values": {"cost": 3, "reliability": 1}},
            {"id": "F", "values": {"cost": 4, "reliability": 0}},
        ],
        "tolerance": 0.0,
    }
    if dataset_id is not None:
        payload["dataset_id"] = dataset_id
    return payload


def test_health_carries_request_identity_and_version():
    response = client.get("/api/v1/health", headers={"X-Request-ID": "req-health"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-health"
    body = response.json()
    assert body["request_id"] == "req-health"
    assert body["status"] == "ok"
    assert body["result"]["algorithm_version"] == "exact-domination-graph-v1"


def test_stateless_sort_returns_exact_hand_calculated_layers_and_evidence():
    response = client.post("/api/v1/sort", json=make_payload())
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["layers"] == [["A", "D"], ["E"], ["C"], ["B"], ["F"]]
    assert result["evidence"]["independent_reference_layers"] == result["layers"]
    assert result["evidence"]["duplicate_groups"] == [["A", "D"]]
    assert result["evidence"]["tolerance_applied_to_ordering"] is False
    assert result["diagnostics"]["uncertainties"] == []
    assert {step["component"] for step in result["processing"]}


def test_dataset_insert_delete_recover_and_request_id_propagate():
    create = client.post(
        "/api/v1/datasets",
        json=make_payload("integration-case"),
        headers={"X-Request-ID": "req-create"},
    )
    assert create.status_code == 201, create.text
    assert create.json()["request_id"] == "req-create"
    assert create.json()["result"]["kernel_version"] == 1

    insert = client.post(
        "/api/v1/datasets/integration-case/candidates:batchInsert",
        json={"candidates": [{"id": "P", "values": {"cost": 0, "reliability": 3}}]},
        headers={"X-Request-ID": "req-insert"},
    )
    assert insert.status_code == 200, insert.text
    inserted = insert.json()["result"]
    assert inserted["layers"] == [["P"], ["A", "D"], ["E"], ["C"], ["B"], ["F"]]
    assert {change["id"] for change in inserted["rank_changes"]} == {
        "A",
        "D",
        "E",
        "C",
        "B",
        "F",
    }

    delete = client.delete(
        "/api/v1/datasets/integration-case/candidates/P",
        headers={"X-Request-ID": "req-delete"},
    )
    assert delete.status_code == 200, delete.text
    deleted = delete.json()["result"]
    assert deleted["layers"] == [["A", "D"], ["E"], ["C"], ["B"], ["F"]]
    assert [change["id"] for change in deleted["rank_changes"]] == [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]

    read = client.get("/api/v1/datasets/integration-case")
    assert read.status_code == 200
    assert read.json()["result"]["layers"] == [["A", "D"], ["E"], ["C"], ["B"], ["F"]]

    remove = client.delete("/api/v1/datasets/integration-case")
    assert remove.status_code == 200
    missing = client.get("/api/v1/datasets/integration-case")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_missing_objective_is_rejected_with_specific_error():
    payload = make_payload()
    del payload["candidates"][0]["values"]["cost"]
    response = client.post("/api/v1/sort", json=payload)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "MISSING_OBJECTIVE"
    assert error["details"]["candidate_id"] == "A"
    assert error["details"]["missing_objectives"] == ["cost"]


def test_invalid_insertion_batch_is_atomic():
    client.post("/api/v1/datasets", json=make_payload("atomic-case"))
    response = client.post(
        "/api/v1/datasets/atomic-case/candidates:batchInsert",
        json={
            "candidates": [
                {"id": "Z", "values": {"cost": 0, "reliability": 4}},
                {"id": "A", "values": {"cost": 0, "reliability": 4}},
            ]
        },
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DUPLICATE_CANDIDATE_ID"

    read = client.get("/api/v1/datasets/atomic-case")
    assert "Z" not in {candidate["id"] for candidate in read.json()["result"]["candidates"]}


def test_unknown_candidate_and_near_equal_uncertainty_are_separate():
    payload = make_payload("uncertainty-case")
    payload["tolerance"] = 0.0
    create = client.post("/api/v1/datasets", json=payload)
    assert create.status_code == 201
    missing = client.delete(
        "/api/v1/datasets/uncertainty-case/candidates/does-not-exist"
    )
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "CANDIDATE_NOT_FOUND"

    near = client.post(
        "/api/v1/datasets/uncertainty-case/candidates:batchInsert",
        json={"candidates": [{"id": "near-A", "values": {"cost": 1, "reliability": 3}}]},
    )
    assert near.status_code == 200
    assert near.json()["warnings"] == []

