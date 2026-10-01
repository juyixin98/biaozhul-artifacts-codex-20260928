from __future__ import annotations

import json

import pytest


def create_model(client, alpha=1.0, beta=1.0):
    response = client.post("/models", json={"alpha": alpha, "beta": beta, "reason": "fixture"})
    assert response.status_code == 201, response.text
    return response.json()


def test_create_update_predict_and_distinct_interval_wording(client, test_logger):
    model = create_model(client, 1.0, 1.0)
    response = client.post(
        f"/models/{model['id']}/batches",
        json={"success_count": 2, "failure_count": 1},
    )
    assert response.status_code == 201

    posterior = client.get(f"/models/{model['id']}/posterior").json()
    prediction = client.post(
        f"/models/{model['id']}/predict",
        json={"trials": 2, "include_distribution": True},
    ).json()

    assert posterior["posterior_alpha"] == 3.0
    assert posterior["posterior_beta"] == 2.0
    assert posterior["credible_interval"]["interpretation"] == "Bayesian credible interval"
    assert (
        posterior["frequentist_confidence_interval"]["interpretation"]
        == "Frequentist confidence interval, not a Bayesian credible interval"
    )
    assert prediction["probabilities"] == pytest.approx([0.2, 0.4, 0.4], abs=1e-12)
    assert abs(prediction["probability_sum"] - 1.0) < 1e-12

    test_logger.note(
        "api_small_predictive",
        "response_contract",
        {
            "model_id": model["id"],
            "posterior": [posterior["posterior_alpha"], posterior["posterior_beta"]],
            "probabilities": prediction["probabilities"],
            "probability_sum": prediction["probability_sum"],
        },
        "API preserved the hand-computed posterior and normalized predictive mass.",
    )


def test_batch_revision_and_recovery_endpoints(client):
    model = create_model(client)
    wrong = client.post(
        f"/models/{model['id']}/batches",
        json={"success_count": 10, "failure_count": 5},
    ).json()
    client.post(
        f"/models/{model['id']}/batches",
        json={"success_count": 1, "failure_count": 1},
    )

    revision = client.post(
        f"/models/{model['id']}/batches/{wrong['id']}/revise",
        json={"success_count": 4, "failure_count": 7},
    )
    assert revision.status_code == 201
    revised_state = client.get(f"/models/{model['id']}").json()
    assert (revised_state["success_count"], revised_state["failure_count"]) == (5, 8)

    recovery = client.post(f"/models/{model['id']}/batches/{wrong['id']}/restore")
    assert recovery.status_code == 200
    recovered_state = client.get(f"/models/{model['id']}").json()
    assert (recovered_state["success_count"], recovered_state["failure_count"]) == (11, 6)

    events = client.get(f"/models/{model['id']}/events").json()
    assert [event["event_type"] for event in events] == [
        "MODEL_CREATED",
        "BATCH_ADDED",
        "BATCH_ADDED",
        "BATCH_REVISION_WITHDREW_ORIGINAL",
        "BATCH_REVISION_ADDED_REPLACEMENT",
        "BATCH_RECOVERY_WITHDREW_REPLACEMENT",
        "BATCH_RECOVERY_RESTORED_ORIGINAL",
    ]


def test_invalid_input_contract_for_prior_and_counts(client):
    response = client.post("/models", json={"alpha": 0.0, "beta": 1.0})
    assert response.status_code == 422
    assert response.json()["error"]["category"] == "invalid_input"

    positive_integer_prior = client.post("/models", json={"alpha": 2, "beta": 1})
    assert positive_integer_prior.status_code == 201
    assert positive_integer_prior.json()["prior_alpha"] == 2.0

    model = create_model(client)
    response = client.post(
        f"/models/{model['id']}/batches",
        json={"success_count": -1, "failure_count": 0},
    )
    assert response.status_code == 422
    assert response.json()["error"]["category"] == "invalid_input"


def test_boolean_integer_rejection_is_not_coerced(client):
    response = client.post(
        "/models",
        json={"alpha": 1.0, "beta": 1.0, "reason": True},
    )
    assert response.status_code == 422

    model = create_model(client)
    response = client.post(
        f"/models/{model['id']}/batches",
        json={"success_count": True, "failure_count": 0},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


def test_not_found_state_conflict_and_resource_exhausted_categories(client):
    missing = client.get("/models/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["error"]["category"] == "not_found"

    model = create_model(client)
    batch = client.post(
        f"/models/{model['id']}/batches",
        json={"success_count": 1, "failure_count": 1},
    ).json()
    client.post(f"/models/{model['id']}/batches/{batch['id']}/revert")
    conflict = client.post(
        f"/models/{model['id']}/batches/{batch['id']}/revert",
        params={"reason": "again"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["category"] == "state_conflict"

    exhausted = client.post(
        f"/models/{model['id']}/predict",
        json={"trials": 10_001, "include_distribution": False},
    )
    assert exhausted.status_code == 413
    assert exhausted.json()["error"]["category"] == "resource_exhausted"


def test_request_id_is_echoed_and_unknown_model_replayable(client):
    response = client.get(
        "/models/missing-replay",
        headers={"x-request-id": "replay-001"},
    )
    assert response.status_code == 404
    assert response.headers["x-request-id"] == "replay-001"
    assert response.json()["error"]["request_id"] == "replay-001"


def test_events_contain_key_intermediate_state_and_judgement(client, run_directory):
    model = create_model(client)
    client.post(
        f"/models/{model['id']}/batches",
        json={"success_count": 3, "failure_count": 2, "reason": "log check"},
    )
    events = client.get(f"/models/{model['id']}/events").json()
    details = events[-1]["details"]
    assert details["intermediate"]["total_success_after_event"] == 3
    assert details["intermediate"]["total_failure_after_event"] == 2
    assert "judgement" in details

    # The test run directory itself should contain replayable JSONL records.
    records = [
        json.loads(line)
        for line in (run_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(record["stage"] == "finished" for record in records)
