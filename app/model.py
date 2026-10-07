"""Model arrays, one SGNS update step, and versioned artifact persistence."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass

import numpy as np

from .errors import ComputationError, ValidationError

ARTIFACT_MAGIC = "local-skipgram-sgns"
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ModelArtifact:
    run_id: str
    tokens: tuple[str, ...]
    counts: np.ndarray
    config: dict
    corpus_fingerprint: str
    input_matrix: np.ndarray
    output_matrix: np.ndarray

    @property
    def dim(self) -> int:
        return self.input_matrix.shape[1]


def initial_embeddings(
    vocab_size: int, dim: int, rng: np.random.Generator, *, init_range: float
) -> tuple[np.ndarray, np.ndarray]:
    input_matrix = rng.uniform(-init_range, init_range, size=(vocab_size, dim)).astype(
        np.float64
    )
    output_matrix = np.zeros((vocab_size, dim), dtype=np.float64)
    return input_matrix, output_matrix


def stable_sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    value = np.asarray(x, dtype=np.float64)
    result = np.piecewise(
        value,
        [value >= 0, value < 0],
        [
            lambda z: 1.0 / (1.0 + np.exp(-z)),
            lambda z: np.exp(z) / (1.0 + np.exp(z)),
        ],
    )
    if value.ndim == 0:
        return float(result)
    return result


def negative_sampling_step(
    input_matrix: np.ndarray,
    output_matrix: np.ndarray,
    *,
    center_id: int,
    positive_id: int,
    negative_ids: list[int],
    learning_rate: float,
) -> dict:
    """Perform one positive-plus-negative SGNS update.

    Return loss and gradients.  Both matrices are updated only after all values
    have been calculated from pre-update snapshots, preventing input/output
    vector updates from overwriting one another.
    """
    if not negative_ids:
        raise ValidationError("at least one negative sample is required")
    if positive_id in negative_ids:
        raise ValidationError(
            "negative ids must not contain the positive target",
            {"positive_id": positive_id, "negative_ids": negative_ids},
        )
    target_ids = np.asarray([positive_id, *negative_ids], dtype=np.int64)
    labels = np.zeros(len(target_ids), dtype=np.float64)
    labels[0] = 1.0

    input_snapshot = np.array(input_matrix[center_id], dtype=np.float64)
    output_snapshot = np.array(output_matrix[target_ids], dtype=np.float64)
    scores = output_snapshot @ input_snapshot
    probabilities = stable_sigmoid(scores)
    upstream = labels - probabilities
    loss = float(
        -np.sum(labels * np.log(np.clip(probabilities, 1e-300, 1.0)))
        - np.sum(
            (1.0 - labels) * np.log(np.clip(1.0 - probabilities, 1e-300, 1.0))
        )
    )

    input_gradient = -learning_rate * (output_snapshot.T @ upstream)
    output_gradients = -learning_rate * np.outer(upstream, input_snapshot)
    if not (
        np.all(np.isfinite(input_snapshot))
        and np.all(np.isfinite(output_snapshot))
        and np.all(np.isfinite(input_gradient))
        and np.all(np.isfinite(output_gradients))
        and np.isfinite(loss)
    ):
        raise ComputationError(
            "skip-gram update produced a non-finite value",
            {"center_id": int(center_id), "positive_id": int(positive_id)},
        )

    input_matrix[center_id] = input_snapshot - input_gradient
    # add.at accumulates repeated negative ids; ordinary advanced indexing
    # would keep only the final duplicate row assignment.
    np.add.at(output_matrix, target_ids, -output_gradients)
    return {
        "loss": loss,
        "scores": [float(x) for x in scores],
        "probabilities": [float(x) for x in probabilities],
        "input_gradient": [float(x) for x in input_gradient],
        "output_gradients": output_gradients.tolist(),
    }


def artifact_to_metadata(artifact: ModelArtifact) -> dict:
    return {
        "magic": ARTIFACT_MAGIC,
        "version": ARTIFACT_VERSION,
        "run_id": artifact.run_id,
        "tokens": list(artifact.tokens),
        "counts": [int(x) for x in artifact.counts],
        "config": artifact.config,
        "corpus_fingerprint": artifact.corpus_fingerprint,
        "input_shape": list(artifact.input_matrix.shape),
        "output_shape": list(artifact.output_matrix.shape),
    }


def save_artifact(path: str, artifact: ModelArtifact) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    metadata = artifact_to_metadata(artifact)
    fd, temporary = tempfile.mkstemp(prefix=".artifact-", dir=directory, suffix=".npz")
    os.close(fd)
    try:
        np.savez(
            temporary,
            input_matrix=artifact.input_matrix,
            output_matrix=artifact.output_matrix,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_artifact(path: str) -> ModelArtifact:
    try:
        with np.load(path, allow_pickle=False) as bundle:
            metadata = json.loads(str(bundle["metadata"]))
            input_matrix = bundle["input_matrix"].astype(np.float64)
            output_matrix = bundle["output_matrix"].astype(np.float64)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationError(
            "model artifact is missing, unreadable, or malformed",
            {"path": path},
        ) from exc

    if metadata.get("magic") != ARTIFACT_MAGIC or metadata.get("version") != ARTIFACT_VERSION:
        raise ValidationError("unsupported model artifact format", {"metadata": metadata})
    tokens = tuple(metadata["tokens"])
    counts = np.asarray(metadata["counts"], dtype=np.int64)
    expected_shape = (len(tokens), metadata["config"]["dim"])
    if input_matrix.shape != expected_shape or output_matrix.shape != expected_shape:
        raise ValidationError(
            "artifact matrix shape does not match frozen vocabulary",
            {"expected": list(expected_shape)},
        )
    if len(tokens) != len(counts) or not np.all(np.isfinite(input_matrix)):
        raise ValidationError("artifact vocabulary or input matrix is invalid")
    if not np.all(np.isfinite(output_matrix)):
        raise ValidationError("artifact output matrix is invalid")
    return ModelArtifact(
        run_id=metadata["run_id"],
        tokens=tokens,
        counts=counts,
        config=metadata["config"],
        corpus_fingerprint=metadata["corpus_fingerprint"],
        input_matrix=input_matrix,
        output_matrix=output_matrix,
    )
