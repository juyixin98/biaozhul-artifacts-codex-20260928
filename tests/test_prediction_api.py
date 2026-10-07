"""Prediction semantics, unknown-word contract, and HTTP error categories."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sgns.app import create_app
from sgns.errors import (
    ComputationFailureError,
    NotFoundError,
)
from sgns.evaluation import cosine_similarity, evaluate_pairs, nearest_neighbours
from sgns.artifact import ModelArtifact
from sgns.config import TrainConfig
from sgns.vocab import build_vocabulary

from .replay_log import RunLogger


DEMO_CORPUS = (
    "london england london england city\n"
    "paris france paris france city\n"
    "berlin germany berlin germany city\n"
)


def _orthogonal_artifact() -> ModelArtifact:
    # Hand-set vectors: x/x synonyms align; unrelated words are orthogonal.
    vocab = build_vocabulary(
        [
            ["london", "england", "city"],
            ["paris", "france", "city"],
        ],
        subsample_t=0.0,
    )
    dim = 2
    vectors = {
        "city": [0.0, 1.0],
        "england": [1.0, 0.2],
        "france": [-1.0, 0.2],
        "london": [1.0, 0.2],
        "paris": [-1.0, 0.2],
    }
    matrix = np.array(
        [vectors[word] for word in vocab.index_to_word],
        dtype=np.float64,
    )
    return ModelArtifact(
        vocab=vocab,
        config=TrainConfig(dim=dim),
        input_vectors=matrix,
        output_vectors=np.zeros_like(matrix),
    )


def test_cosine_and_neighbours_with_exact_geometry() -> None:
    log = RunLogger("exact_geometry")
    artifact = _orthogonal_artifact()
    same = cosine_similarity(artifact, "london", "england")
    opposite_family = cosine_similarity(artifact, "london", "france")
    self_sim = cosine_similarity(artifact, "city", "city")
    assert same["similarity"] == pytest.approx(1.0, abs=1e-12)
    assert self_sim["similarity"] == pytest.approx(1.0, abs=1e-12)
    assert opposite_family["similarity"] < 0.0
    neighbours = nearest_neighbours(artifact, "london", top_k=3)
    words = [item["word"] for item in neighbours["neighbours"]]
    assert words[0] == "england"
    assert "london" not in words
    assert len(words) <= 3
    log.record(
        "geometry_contract",
        "identical-direction pairs are cosine 1, opposite-sign family is "
        "negative, self similarity is 1 and nearest neighbours exclude self",
        expected={"london_england": 1.0, "top_neighbour": "england"},
        observed={
            "london_england": same["similarity"],
            "london_france": opposite_family["similarity"],
            "self": self_sim["similarity"],
            "neighbours": words,
        },
    )


def test_zero_vector_similarity_is_computation_failure() -> None:
    log = RunLogger("zero_vector")
    vocab = build_vocabulary([["a", "b"]], subsample_t=0.0)
    artifact = ModelArtifact(
        vocab=vocab,
        config=TrainConfig(dim=2),
        input_vectors=np.array([[0.0, 0.0], [1.0, 0.0]]),
        output_vectors=np.zeros((2, 2)),
    )
    with pytest.raises(ComputationFailureError) as exc:
        cosine_similarity(artifact, "a", "b")
    assert exc.value.code == "COMPUTATION_FAILED"
    log.record(
        "zero_norm_category",
        "cosine against a zero-norm vector is COMPUTATION_FAILED rather than "
        "NaN or a misleading 0.0",
        expected="COMPUTATION_FAILED",
        observed=exc.value.code,
    )


def test_unknown_word_is_not_found_contract() -> None:
    log = RunLogger("unknown_word")
    artifact = _orthogonal_artifact()
    with pytest.raises(NotFoundError) as exc:
        cosine_similarity(artifact, "london", "atlantis")
    assert exc.value.code == "NOT_FOUND"
    assert exc.value.details["word"] == "atlantis"
    log.record(
        "unknown_word",
        "querying a word outside the frozen vocabulary is NOT_FOUND and names "
        "the offending word",
        expected="NOT_FOUND",
        observed={"code": exc.value.code, "word": exc.value.details["word"]},
    )


def test_evaluate_pairs_uses_only_caller_labels() -> None:
    log = RunLogger("fixture_evaluation")
    artifact = _orthogonal_artifact()
    pairs = [
        {"word_a": "london", "word_b": "england", "label": 1},
        {"word_a": "london", "word_b": "france", "label": 0},
    ]
    result = evaluate_pairs(artifact, pairs)
    assert result["pair_count"] == 2
    assert result["agreements"] == 2
    assert result["agreement"] == 1.0
    assert "not a semantic understanding benchmark" in result["note"]
    with pytest.raises(Exception):
        evaluate_pairs(artifact, [{"word_a": "london", "word_b": "england"}])
    log.record(
        "caller_labels_only",
        "evaluation reports agreement with caller-provided synthetic labels and "
        "explicitly disclaims any real semantic benchmark",
        expected={"agreements": 2, "agreement": 1.0},
        observed={
            "agreements": result["agreements"],
            "agreement": result["agreement"],
            "note": result["note"],
        },
    )


def test_http_end_to_end_and_error_status_mapping(tmp_path) -> None:
    log = RunLogger("http_end_to_end")
    client = TestClient(
        create_app(
            db_path=tmp_path / "db.sqlite",
            artifact_dir=tmp_path / "art",
        )
    )
    assert client.get("/health").json() == {"status": "ok"}

    created = client.post(
        "/corpora",
        json={"corpus_id": "geo", "text": DEMO_CORPUS},
    )
    assert created.status_code == 201
    assert created.json()["tokens"] == 15

    trained = client.post(
        "/training-runs",
        json={
            "run_id": "geo-run",
            "corpus_id": "geo",
            "config": {
                "dim": 8,
                "epochs": 20,
                "window": 2,
                "negatives": 3,
                "subsample_t": 0.0,
                "seed": 42,
            },
        },
    )
    assert trained.status_code == 201, trained.text
    assert trained.json()["status"] == "COMPLETED"

    vector = client.get("/training-runs/geo-run/vectors/london")
    assert vector.status_code == 200
    assert len(vector.json()["vector"]) == 8

    similarity = client.post(
        "/training-runs/geo-run/similarity",
        json={"word_a": "london", "word_b": "england"},
    )
    assert similarity.status_code == 200
    score = similarity.json()["similarity"]
    assert -1.0 <= score <= 1.0

    neighbours = client.post(
        "/training-runs/geo-run/neighbours",
        json={"word": "london", "top_k": 3},
    )
    assert neighbours.status_code == 200
    neighbour_words = [n["word"] for n in neighbours.json()["neighbours"]]
    assert "london" not in neighbour_words
    assert len(neighbour_words) == 3

    evaluated = client.post(
        "/training-runs/geo-run/evaluate",
        json={
            "pairs": [
                {"word_a": "london", "word_b": "england", "label": 1},
                {"word_a": "paris", "word_b": "london", "label": 0},
            ]
        },
    )
    assert evaluated.status_code == 200

    # --- Distinct failure categories over HTTP ------------------------------
    unknown = client.get("/training-runs/geo-run/vectors/xylophone")
    assert unknown.status_code == 404
    assert unknown.json()["error"] == "NOT_FOUND"

    missing_run = client.get("/training-runs/no-such-run/vectors/london")
    assert missing_run.status_code == 404
    assert missing_run.json()["error"] == "NOT_FOUND"

    duplicate = client.post(
        "/training-runs",
        json={"run_id": "geo-run", "corpus_id": "geo", "config": {}},
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"] == "STATE_CONFLICT"

    bad_config = client.post(
        "/training-runs",
        json={
            "run_id": "bad-cfg",
            "corpus_id": "geo",
            "config": {"dim": -3},
        },
    )
    assert bad_config.status_code == 400
    assert bad_config.json()["error"] == "INPUT_ERROR"

    empty_corpus = client.post(
        "/corpora",
        json={"corpus_id": "empty", "text": "   "},
    )
    assert empty_corpus.status_code == 400
    assert empty_corpus.json()["error"] == "INPUT_ERROR"

    over_budget = client.post(
        "/corpora",
        json={"corpus_id": "over", "text": "a b c d", "max_tokens": 2},
    )
    assert over_budget.status_code == 413
    assert over_budget.json()["error"] == "RESOURCE_EXHAUSTED"

    log.record(
        "http_status_mapping",
        "happy path trains and serves vectors/similarity/neighbours/evaluate; "
        "404 NOT_FOUND, 409 STATE_CONFLICT, 400 INPUT_ERROR, 413 "
        "RESOURCE_EXHAUSTED are all distinguishable by error code",
        expected={
            "unknown_word": 404,
            "missing_run": 404,
            "duplicate_run": 409,
            "bad_config": 400,
            "empty_corpus": 400,
            "over_budget": 413,
        },
        observed={
            "unknown_word": unknown.status_code,
            "missing_run": missing_run.status_code,
            "duplicate_run": duplicate.status_code,
            "bad_config": bad_config.status_code,
            "empty_corpus": empty_corpus.status_code,
            "over_budget": over_budget.status_code,
            "similarity_score": score,
            "neighbours": neighbour_words,
        },
    )


def test_http_computation_failure_category(tmp_path) -> None:
    log = RunLogger("http_computation_failure")
    client = TestClient(
        create_app(
            db_path=tmp_path / "db.sqlite",
            artifact_dir=tmp_path / "art",
        )
    )
    client.post("/corpora", json={"corpus_id": "c", "text": "a b c a b c"})
    failed = client.post(
        "/training-runs",
        json={
            "run_id": "boom",
            "corpus_id": "c",
            "config": {"lr": 1.0e308, "epochs": 1, "dim": 2},
        },
    )
    assert failed.status_code == 422, failed.text
    assert failed.json()["error"] == "COMPUTATION_FAILED"
    # Failed runs remain recorded and querying them is a state conflict.
    run_state = client.get("/training-runs/boom")
    assert run_state.status_code == 200
    assert run_state.json()["status"] == "FAILED"
    query = client.get("/training-runs/boom/vectors/a")
    assert query.status_code == 409
    assert query.json()["error"] == "STATE_CONFLICT"
    log.record(
        "computation_failure_persisted",
        "non-finite training math maps to 422 COMPUTATION_FAILED; the run is "
        "persisted FAILED and cannot be queried (409 STATE_CONFLICT)",
        expected={"train": 422, "state": "FAILED", "query": 409},
        observed={
            "train": failed.status_code,
            "error": failed.json()["error"],
            "state": run_state.json()["status"],
            "query": query.status_code,
        },
    )
