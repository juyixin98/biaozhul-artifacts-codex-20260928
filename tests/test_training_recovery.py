"""Fixed-corpus deterministic replay, artifact recovery, and queries."""
from __future__ import annotations

import numpy as np
import pytest

from app.config import TrainConfig
from app.data import corpus_fingerprint
from app.errors import ValidationError
from app.model import load_artifact
from app.prediction import cosine_similarity, most_similar, vector_for
from app.storage import ModelStore
from app.training import train_skipgram

FIXTURE_CORPUS = [
    ["alpha", "beta", "alpha", "beta"],
    ["gamma", "alpha", "gamma"],
]
FIXED_CONFIG = TrainConfig(
    seed=4242,
    epochs=3,
    window=1,
    negatives=2,
    dim=4,
    learning_rate=0.05,
    subsample_t=1.0,
    min_count=1,
    init_range=0.25,
)


def train_once():
    return train_skipgram(
        FIXTURE_CORPUS, FIXED_CONFIG, run_id="replay-fixed", event_listener=None
    )


def test_same_corpus_and_seed_replays_all_matrices_and_events(evidence):
    first = train_once()
    second = train_once()

    assert np.array_equal(first.artifact.input_matrix, second.artifact.input_matrix)
    assert np.array_equal(first.artifact.output_matrix, second.artifact.output_matrix)
    assert first.events == second.events
    assert first.final_loss == second.final_loss
    assert np.all(np.isfinite(first.artifact.input_matrix))
    assert np.all(np.isfinite(first.artifact.output_matrix))

    first_steps = [event for event in first.events if event["type"] == "step"]
    first_epoch = [event for event in first.events if event["type"] == "epoch_completed"]
    assert len(first_steps) == sum(event["pair_count"] for event in first_epoch)
    assert first.events[0]["run_id"] == "replay-fixed"
    assert first.events[0]["seed"] == 4242
    assert first.events[0]["corpus_fingerprint"] == corpus_fingerprint(FIXTURE_CORPUS)
    initialized = [event for event in first.events if event["type"] == "vectors_initialized"]
    assert len(initialized) == 1
    assert initialized[0]["input_rows"] == 3
    assert "init_rng_after_initialization" in initialized[0]
    step_events = [event for event in first.events if event["type"] == "step"]
    assert all("rejected_negative_candidates" in event for event in step_events)
    evidence(
        "fixed_corpus_replay",
        {"matrix_equal": True, "events_equal": True, "step_count": len(first_steps)},
        {
            "matrix_equal": bool(
                np.array_equal(first.artifact.input_matrix, second.artifact.input_matrix)
                and np.array_equal(
                    first.artifact.output_matrix, second.artifact.output_matrix
                )
            ),
            "events_equal": first.events == second.events,
            "step_count": len(first_steps),
        },
        "same corpus, root seed, and config must reproduce ordered examples and numeric updates",
    )


def test_input_and_output_matrices_receive_distinct_updates(evidence):
    result = train_once()
    assert not np.allclose(result.artifact.input_matrix, 0.0)
    assert not np.allclose(result.artifact.output_matrix, 0.0)
    assert not np.allclose(
        result.artifact.input_matrix, result.artifact.output_matrix
    )
    evidence(
        "separate_embedding_tables",
        "nonzero and unequal input/output matrices",
        {
            "input_norm": float(np.linalg.norm(result.artifact.input_matrix)),
            "output_norm": float(np.linalg.norm(result.artifact.output_matrix)),
        },
        "center and context gradients update two distinct tables",
    )


def test_sqlite_persistence_recovers_artifact_and_unknown_query_is_validation(tmp_path, evidence):
    result = train_once()
    with ModelStore(str(tmp_path / "runs.sqlite"), str(tmp_path / "artifacts")) as store:
        path = store.save_training_run(result, FIXTURE_CORPUS)
        recovered_artifact = store.load_artifact("replay-fixed")
        loaded_directly = load_artifact(path)

        assert np.array_equal(
            recovered_artifact.input_matrix, result.artifact.input_matrix
        )
        assert np.array_equal(
            loaded_directly.output_matrix, result.artifact.output_matrix
        )
        assert len(store.get_events("replay-fixed")) == len(result.events)

        alpha_vector = vector_for(recovered_artifact, "alpha")
        assert alpha_vector.shape == (4,)
        assert cosine_similarity(recovered_artifact, "alpha", "alpha") == pytest.approx(1.0)
        neighbors = most_similar(recovered_artifact, "alpha", k=2)
        assert {item["token"] for item in neighbors} == {"beta", "gamma"}
        with pytest.raises(ValidationError) as excinfo:
            vector_for(recovered_artifact, "nope")
        assert excinfo.value.category == "validation_error"
        evidence(
            "sqlite_artifact_recovery_unknown_token",
            {
                "recovered_shape": [3, 4],
                "self_similarity": 1.0,
                "unknown_category": "validation_error",
            },
            {
                "recovered_shape": list(recovered_artifact.input_matrix.shape),
                "self_similarity": cosine_similarity(recovered_artifact, "alpha", "alpha"),
                "unknown_category": excinfo.value.category,
            },
            "recovery must restore exact vectors and OOV must remain an explicit query failure",
        )


def test_duplicate_run_id_is_state_conflict(tmp_path):
    result = train_once()
    with ModelStore(str(tmp_path / "runs.sqlite"), str(tmp_path / "artifacts")) as store:
        store.save_training_run(result, FIXTURE_CORPUS)
        with pytest.raises(Exception) as excinfo:
            store.save_training_run(result, FIXTURE_CORPUS)
        assert excinfo.value.category == "state_conflict"
