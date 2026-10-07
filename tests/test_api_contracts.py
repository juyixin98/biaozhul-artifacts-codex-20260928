"""HTTP-level data/error contracts."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.storage import ModelStore


CORPUS = [
    ["alpha", "beta", "alpha"],
    ["beta", "gamma", "beta"],
]


@pytest.fixture
def client(tmp_path):
    with ModelStore(str(tmp_path / "runs.sqlite"), str(tmp_path / "artifacts")) as store:
        with TestClient(create_app(store)) as test_client:
            yield test_client


def train(client, **overrides):
    payload = {
        "corpus": CORPUS,
        "seed": 31337,
        "epochs": 2,
        "window": 1,
        "negatives": 2,
        "dim": 4,
        "learning_rate": 0.05,
        "subsample_t": 1.0,
        "min_count": 1,
        "init_range": 0.1,
    }
    payload.update(overrides)
    response = client.post("/train", json=payload)
    assert response.status_code == 201, response.text
    return response.json()["run_id"]


def test_training_vector_similarity_neighbors_and_events(client):
    run_id = train(client)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    vector = client.get(f"/vectors/{run_id}", params={"token": "alpha"})
    assert vector.status_code == 200
    assert len(vector.json()["vector"]) == 4

    similarity = client.get(
        f"/similarity/{run_id}", params={"token_a": "alpha", "token_b": "alpha"}
    )
    assert similarity.status_code == 200
    assert similarity.json()["similarity"] == pytest.approx(1.0)

    neighbors = client.get(
        f"/neighbors/{run_id}", params={"token": "alpha", "k": 2}
    )
    assert neighbors.status_code == 200
    assert {item["token"] for item in neighbors.json()["neighbors"]} == {"beta", "gamma"}

    events = client.get(f"/runs/{run_id}/events")
    assert events.status_code == 200
    assert events.json()["event_count"] > 0
    assert events.json()["events"][0]["corpus_fingerprint"]

    evaluation = client.post(
        f"/evaluations/{run_id}", json={"pairs": [["alpha", "beta"]]}
    )
    assert evaluation.status_code == 201
    assert evaluation.json()["metrics"]["kind"] == "synthetic_cooccurrence_rank"


def test_unknown_run_is_state_conflict_and_unknown_token_is_validation(client):
    missing_run = client.get("/vectors/missing", params={"token": "alpha"})
    assert missing_run.status_code == 409
    assert missing_run.json()["error"] == "state_conflict"

    run_id = train(client)
    unknown_token = client.get(f"/vectors/{run_id}", params={"token": "oov"})
    assert unknown_token.status_code == 400
    assert unknown_token.json()["error"] == "validation_error"
    assert unknown_token.json()["details"]["token"] == "oov"


def test_malformed_corpus_has_validation_category(client, evidence):
    response = client.post("/train", json={"corpus": [["ok"], []]})
    assert response.status_code == 400
    assert response.json()["error"] == "validation_error"
    assert response.json()["details"]["sentence_index"] == 1

    token_response = client.post(
        "/train",
        json={"corpus": [["1bad"]], "seed": 1, "epochs": 1, "subsample_t": 1.0},
    )
    assert token_response.status_code == 400
    body = token_response.json()
    assert body["error"] == "validation_error"
    assert body["details"]["sentence_index"] == 0
    evidence(
        "http_input_error_category",
        "400 validation_error for malformed token",
        {"status": token_response.status_code, "body": body},
        "HTTP clients need to distinguish bad data from server state and numeric failures",
    )


def test_persisted_failed_run_records_computation_category(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated non-finite kernel")

    monkeypatch.setattr("app.api.train_skipgram", boom)
    response = client.post(
        "/train",
        json={
            "corpus": CORPUS,
            "seed": 7,
            "epochs": 1,
            "window": 1,
            "negatives": 1,
            "dim": 2,
            "subsample_t": 1.0,
        },
    )
    assert response.status_code == 500
    assert response.json()["error"] == "computation_failed"

    run_id = response_request_run_id(response)
    failed = client.get(f"/runs/{run_id}")
    assert failed.status_code == 200
    assert failed.json()["status"] == "failed"
    assert failed.json()["error_category"] == "computation_failed"
    artifact = client.get(f"/vectors/{run_id}", params={"token": "alpha"})
    assert artifact.status_code == 409


def test_resource_exhaustion_is_503_and_persisted(client, monkeypatch):
    from app.errors import ResourceExhaustedError

    def exhausted(*args, **kwargs):
        raise ResourceExhaustedError("simulated budget exhausted")

    monkeypatch.setattr("app.api.train_skipgram", exhausted)
    response = client.post(
        "/train",
        json={
            "corpus": CORPUS,
            "seed": 8,
            "epochs": 1,
            "window": 1,
            "negatives": 1,
            "dim": 2,
            "subsample_t": 1.0,
        },
    )
    assert response.status_code == 503
    assert response.json()["error"] == "resource_exhausted"
    run_id = response_request_run_id(response)
    assert client.get(f"/runs/{run_id}").json()["error_category"] == "resource_exhausted"


def response_request_run_id(response):
    """Recover the deterministic run id from the originating request."""
    from app.config import TrainConfig
    from app.data import corpus_fingerprint, validate_corpus
    from app.training import make_run_id

    request = response.request
    import json

    payload = json.loads(request.content)
    corpus = validate_corpus(payload["corpus"])
    payload.pop("corpus")
    config = TrainConfig.from_dict(payload)
    return make_run_id(corpus, config)
