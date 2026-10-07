"""Word-vector lookup, pair similarity, and nearest-neighbor queries."""
from __future__ import annotations

import numpy as np

from .errors import ComputationError, ValidationError
from .model import ModelArtifact


def token_id(artifact: ModelArtifact, token: str) -> int:
    if not isinstance(token, str) or token not in artifact.tokens:
        raise ValidationError(
            "unknown token; it is absent from the frozen vocabulary",
            {"token": token if isinstance(token, str) else None},
        )
    return artifact.tokens.index(token)


def vector_for(artifact: ModelArtifact, token: str) -> np.ndarray:
    return artifact.input_matrix[token_id(artifact, token)].copy()


def cosine_similarity(artifact: ModelArtifact, token_a: str, token_b: str) -> float:
    left = vector_for(artifact, token_a)
    right = vector_for(artifact, token_b)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not np.isfinite(denominator) or denominator == 0.0:
        raise ComputationError(
            "cosine similarity is undefined for a zero-norm vector",
            {"token_a": token_a, "token_b": token_b},
        )
    return float(np.dot(left, right) / denominator)


def most_similar(
    artifact: ModelArtifact, token: str, *, k: int = 5
) -> list[dict]:
    query_id = token_id(artifact, token)
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValidationError("k must be an integer", {"actual": repr(k)})
    maximum = artifact.input_matrix.shape[0] - 1
    if not 1 <= k <= maximum:
        raise ValidationError(
            "k must be in [1, vocabulary_size - 1]",
            {"actual": k, "max": maximum},
        )

    query = artifact.input_matrix[query_id]
    query_norm = float(np.linalg.norm(query))
    if not np.isfinite(query_norm) or query_norm == 0.0:
        raise ComputationError(
            "nearest-neighbor query is undefined for a zero-norm query vector",
            {"token": token},
        )
    norms = np.linalg.norm(artifact.input_matrix, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
        raise ComputationError("vocabulary contains a zero-norm vector")
    scores = (artifact.input_matrix @ query) / (norms * query_norm)
    scores[query_id] = -np.inf
    order = np.argsort(-scores, kind="stable")[:k]
    return [
        {"token": artifact.tokens[int(i)], "similarity": float(scores[int(i)])}
        for i in order
    ]
