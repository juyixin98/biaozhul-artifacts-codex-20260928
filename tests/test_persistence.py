"""Artifact integrity, SQLite lifecycle, replay and query error contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sgns.artifact import ModelArtifact, load_artifact, save_artifact
from sgns.config import TrainConfig
from sgns.errors import (
    ComputationFailureError,
    NotFoundError,
    StateConflictError,
)
from sgns.service import EmbeddingService
from sgns.storage import Store
from sgns.vocab import build_vocabulary

from .replay_log import RunLogger


SYNTHETIC_CORPUS = (
    "alpha beta alpha gamma alpha beta\n"
    "gamma alpha beta delta gamma\n"
    "beta alpha gamma beta alpha\n"
)


def _make_artifact(seed: int = 11) -> ModelArtifact:
    # Small deterministic vectors with independent vocab/config fixtures.
    vocab = build_vocabulary(
        [["alpha", "beta"], ["gamma", "delta"]],
        subsample_t=0.0,
    )
    config = TrainConfig(dim=3, seed=seed)
    rng = np.random.default_rng(seed)
    input_vectors = rng.normal(size=(vocab.size, 3))
    output_vectors = rng.normal(size=(vocab.size, 3)) * 0.1
    return ModelArtifact(vocab, config, input_vectors, output_vectors)


def test_artifact_roundtrip_and_digest_mismatch(tmp_path) -> None:
    log = RunLogger("artifact_integrity")
    artifact = _make_artifact()
    saved = save_artifact(tmp_path, "run-a", artifact)
    reloaded = load_artifact(tmp_path, "run-a")
    assert np.array_equal(reloaded.input_vectors, artifact.input_vectors)
    assert np.array_equal(reloaded.output_vectors, artifact.output_vectors)
    assert reloaded.vocab.index_to_word == artifact.vocab.index_to_word
    assert reloaded.config.seed == artifact.config.seed

    # Tamper the archive without updating the sidecar digest.
    tampered = tmp_path / "run-a.npz"
    data = bytearray(tampered.read_bytes())
    data[200] ^= 0xFF
    tampered.write_bytes(bytes(data))
    with pytest.raises(ComputationFailureError) as exc:
        load_artifact(tmp_path, "run-a")
    assert exc.value.code == "COMPUTATION_FAILED"
    assert exc.value.details["expected"] != exc.value.details["actual"]
    log.record(
        "digest_guards_vectors",
        "round-trip preserves vectors/config; a mutated archive is rejected "
        "as COMPUTATION_FAILED because the sha256 sidecar no longer matches",
        expected="digest mismatch failure",
        observed={
            "code": exc.value.code,
            "expected_prefix": exc.value.details["expected"][:12],
            "actual_prefix": exc.value.details["actual"][:12],
            "saved_sha256": saved["sha256"][:12],
        },
    )


def test_artifact_shape_mismatch_is_computation_failure() -> None:
    log = RunLogger("artifact_shape")
    vocab = build_vocabulary([["a", "b"]], subsample_t=0.0)
    config = TrainConfig(dim=4)
    with pytest.raises(ComputationFailureError) as exc:
        ModelArtifact(
            vocab=vocab,
            config=config,
            input_vectors=np.zeros((3, 4)),
            output_vectors=np.zeros((3, 4)),
        )
    assert exc.value.code == "COMPUTATION_FAILED"
    log.record(
        "shape_contract",
        "vectors must be (vocab_size, dim); mismatch is COMPUTATION_FAILED",
        expected=[2, 4],
        observed=exc.value.details["observed"],
    )


def test_repeated_training_is_identical_and_recoverable(tmp_path) -> None:
    log = RunLogger("repeated_training")
    db_one = tmp_path / "one.db"
    db_two = tmp_path / "two.db"
    art_one = tmp_path / "art_one"
    art_two = tmp_path / "art_two"
    config = {
        "dim": 6,
        "epochs": 4,
        "window": 2,
        "negatives": 3,
        "subsample_t": 1e-3,
        "seed": 20260,
        "min_count": 1,
    }

    svc_a = EmbeddingService(db_one, art_one)
    svc_a.create_corpus("corpus", SYNTHETIC_CORPUS)
    run_a = svc_a.start_run("run", "corpus", config, record_events=3)

    # Independent store/artifact directories reproduce bit-identical vectors.
    svc_b = EmbeddingService(db_two, art_two)
    svc_b.create_corpus("corpus", SYNTHETIC_CORPUS)
    run_b = svc_b.start_run("run", "corpus", config)

    artifact_a = load_artifact(art_one, "run")
    artifact_b = load_artifact(art_two, "run")
    assert np.array_equal(
        artifact_a.input_vectors,
        artifact_b.input_vectors,
    )
    assert np.array_equal(
        artifact_a.output_vectors,
        artifact_b.output_vectors,
    )
    assert run_a["stats"] == run_b["stats"]

    # Restarting an existing run id is a state conflict, not a silent retrain.
    with pytest.raises(StateConflictError) as conflict:
        svc_a.start_run("run", "corpus", config)
    assert conflict.value.code == "STATE_CONFLICT"
    assert conflict.value.details["status"] == "COMPLETED"

    # Recovery from SQLite: status/digest/artifact all line up.
    row = Store(db_one).get_run("run")
    assert row["status"] == "COMPLETED"
    assert row["artifact_sha256"] == run_a["artifact"]["sha256"]
    log.record(
        "deterministic_repeat_and_recovery",
        "same corpus+config+seed in separate stores produces identical input "
        "and output vectors and stats; duplicate run id is STATE_CONFLICT; "
        "SQLite records COMPLETED with the artifact digest",
        expected="identical vectors and stats; COMPLETED row",
        observed={
            "input_equal": bool(
                np.array_equal(artifact_a.input_vectors, artifact_b.input_vectors)
            ),
            "output_equal": bool(
                np.array_equal(
                    artifact_a.output_vectors, artifact_b.output_vectors
                )
            ),
            "stats": run_a["stats"],
            "duplicate_run": conflict.value.code,
            "recovered_status": row["status"],
            "digest_match": row["artifact_sha256"]
            == run_a["artifact"]["sha256"],
        },
        config=config,
    )


def test_trace_records_fixed_first_steps(tmp_path) -> None:
    log = RunLogger("trace_replay")
    svc = EmbeddingService(tmp_path / "db.sqlite", tmp_path / "art")
    svc.create_corpus("c", "a b c d\nd c b a")
    config = {
        "dim": 4,
        "epochs": 1,
        "window": 1,
        "negatives": 1,
        "subsample_t": 0.0,
        "seed": 5,
    }
    run = svc.start_run("r", "c", config, record_events=2)
    trace = run["trace"]
    assert len(trace) == 2
    first = trace[0]
    assert first["epoch"] == 0
    assert first["sentence_index"] == 0
    assert first["center_position"] == 0
    assert 1 <= first["dynamic_window"] <= 1  # window fixed to 1
    assert first["positive_id"] != first["negative_ids"][0]
    assert len(first["input_vector_before"]) == 4
    assert len(first["input_vector_after"]) == 4
    assert first["coefficients"][0] > 0
    assert first["coefficients"][1] < 0
    log.record(
        "trace_contract",
        "first event is epoch 0, sentence 0, center 0; positive coefficient "
        "is positive and negative coefficient is negative; vectors recorded "
        "before and after",
        expected={"epoch": 0, "sentence": 0, "center": 0},
        observed={
            "epoch": first["epoch"],
            "sentence_index": first["sentence_index"],
            "center_position": first["center_position"],
            "positive_coefficient": first["coefficients"][0],
            "negative_coefficient": first["coefficients"][1],
        },
    )
