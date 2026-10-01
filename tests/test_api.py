from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def create_problem(problem_id="api-case", candidates=None, numeric_settings=None):
    payload = {
        "problem_id": problem_id,
        "objectives": [
            {"name": "cost", "direction": "min"},
            {"name": "gain", "direction": "max"},
        ],
        "candidates": candidates or [],
    }
    if numeric_settings is not None:
        payload["numeric_settings"] = numeric_settings
    return client.post("/v1/problems", json=payload, headers={"X-Request-ID": "req-create"})


def test_create_problem_and_read_explicit_hand_computed_layers():
    response = create_problem(
        "layers",
        [
            {"id": "a", "objectives": [1, 1]},
            {"id": "b", "objectives": [2, 2]},
            {"id": "c", "objectives": [1, 1]},
            {"id": "d", "objectives": [3, 1]},
        ],
    )
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-create"
    body = response.json()
    assert body["request_id"] == "req-create"
    assert body["data"]["state"]["layers"] == [
        {"rank": 1, "candidate_ids": ["a", "b", "c"]},
        {"rank": 2, "candidate_ids": ["d"]},
    ]
    evidence = body["evidence"]
    assert evidence["algorithm_version"] == "exact-nsga-nds/1.0.0"
    assert evidence["processing_stage"] == "initial-full-comparison"
    assert evidence["duplicate_groups"] == [["a", "c"]]
    assert all(check["passed"] for check in evidence["checks"])
    assert body["uncertainty"] == []


def test_insert_and_delete_incrementally_update_layers():
    create_response = create_problem(
        "increment",
        [
            {"id": "top", "objectives": [0, 0]},
            {"id": "middle", "objectives": [2, -2]},
            {"id": "bottom", "objectives": [3, -3]},
        ],
    )
    assert create_response.status_code == 200
    initial = create_response.json()["data"]["state"]
    assert initial["layers"] == [
        {"rank": 1, "candidate_ids": ["top"]},
        {"rank": 2, "candidate_ids": ["middle"]},
        {"rank": 3, "candidate_ids": ["bottom"]},
    ]

    insert = client.post(
        "/v1/problems/increment/candidates",
        json={"candidates": [{"id": "x", "objectives": [1, -1]}]},
        headers={"X-Request-ID": "req-insert"},
    )
    inserted = insert.json()
    assert insert.status_code == 200
    assert inserted["request_id"] == "req-insert"
    assert inserted["data"]["candidate_id"] == "x"
    assert inserted["evidence"]["processing_stage"] == "incremental-local-rerank"
    assert inserted["data"]["state"]["layers"] == [
        {"rank": 1, "candidate_ids": ["top"]},
        {"rank": 2, "candidate_ids": ["x"]},
        {"rank": 3, "candidate_ids": ["middle"]},
        {"rank": 4, "candidate_ids": ["bottom"]},
    ]

    delete = client.delete(
        "/v1/problems/increment/candidates/x",
        headers={"X-Request-ID": "req-delete"},
    )
    deleted = delete.json()
    assert delete.status_code == 200
    assert deleted["request_id"] == "req-delete"
    assert deleted["evidence"]["processing_stage"] == "incremental-downstream-restore"
    assert deleted["data"]["state"]["layers"] == [
        {"rank": 1, "candidate_ids": ["top"]},
        {"rank": 2, "candidate_ids": ["middle"]},
        {"rank": 3, "candidate_ids": ["bottom"]},
    ]
    assert all(check["passed"] for check in deleted["evidence"]["checks"])


def test_missing_objective_is_a_specific_failure():
    create_problem("bad")
    response = client.post(
        "/v1/problems/bad/candidates",
        json={"candidates": [{"id": "bad", "objectives": [1]}]},
        headers={"X-Request-ID": "req-bad"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["request_id"] == "req-bad"
    assert body["errors"][0]["code"] == "objective_count_mismatch"
    assert body["errors"][0]["stage"] == "model-parsing"


def test_candidate_relations_and_pairwise_explanation_are_readable():
    create_problem(
        "explain",
        [
            {"id": "a", "objectives": [1, 1]},
            {"id": "b", "objectives": [3, 1]},
        ],
    )
    relation = client.get(
        "/v1/problems/explain/candidates/a",
        headers={"X-Request-ID": "req-relation"},
    )
    assert relation.status_code == 200
    assert relation.json()["data"]["dominates"] == ["b"]

    explanation = client.get("/v1/problems/explain/explain/a/b")
    assert explanation.status_code == 200
    data = explanation.json()["data"]
    assert data["relation"] == "dominates"
    assert [dimension["direction"] for dimension in data["dimensions"]] == [
        "min",
        "max",
    ]
    assert data["dimensions"][0]["comparison"] == "left_better"
    assert data["dimensions"][1]["comparison"] == "equal"


def test_quantized_mode_separates_uncertainty_from_errors():
    response = create_problem(
        "quantized",
        [
            {"id": "a", "objectives": ["0", "0"]},
            {"id": "b", "objectives": ["0.05", "0.05"]},
        ],
        numeric_settings={"mode": "quantized", "tolerance": "0.1"},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["evidence"]["uncertainty"]
    assert body["errors"] == []
    assert body["data"]["state"]["numeric_policy"]["exact_by_default"] is False


def test_unknown_problem_is_not_found_error():
    response = client.get("/v1/problems/missing")
    assert response.status_code == 404
    assert response.json()["errors"][0]["code"] == "problem_not_found"
