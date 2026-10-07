"""Prediction and fixture evaluation over frozen input word vectors.

Only *input* vectors are served as word vectors (the output vectors remain
in the artifact for audit and research). Cosine similarity is computed with
float64. A zero-norm row cannot define a direction and raises
``COMPUTATION_FAILED`` rather than silently returning NaN or zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .artifact import ModelArtifact
from .errors import ComputationFailureError, InputError, NotFoundError


def _row(artifact: ModelArtifact, word: str) -> tuple[int, np.ndarray]:
    if not isinstance(word, str) or not word:
        raise InputError("word must be a non-empty string")
    if not artifact.vocab.contains(word):
        raise NotFoundError("word is not in the frozen vocabulary", word=word)
    word_id = artifact.vocab.index(word)
    return word_id, artifact.input_vectors[word_id]


def get_vector(artifact: ModelArtifact, word: str) -> dict[str, Any]:
    word_id, vector = _row(artifact, word)
    if not np.isfinite(vector).all():
        raise ComputationFailureError(
            "stored vector contains non-finite values",
            word=word,
        )
    return {
        "word": word,
        "word_id": word_id,
        "vector": vector.astype(float).tolist(),
        "norm": float(np.linalg.norm(vector)),
    }


def _normalized(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if norm == 0.0 or not np.isfinite(norm):
        raise ComputationFailureError(
            "cannot compute cosine for a zero or non-finite vector",
            word=label,
            norm=norm,
        )
    return vector / norm


def cosine_similarity(
    artifact: ModelArtifact,
    word_a: str,
    word_b: str,
) -> dict[str, Any]:
    id_a, vec_a = _row(artifact, word_a)
    id_b, vec_b = _row(artifact, word_b)
    unit_a = _normalized(vec_a, word_a)
    unit_b = _normalized(vec_b, word_b)
    score = float(np.dot(unit_a, unit_b))
    # Cosine sits in [-1, 1]; clamp only round-off and report raw range issues.
    if abs(score) > 1.0 + 1e-9 or not np.isfinite(score):
        raise ComputationFailureError(
            "cosine score left the valid range",
            score=score,
        )
    score = max(-1.0, min(1.0, score))
    return {
        "word_a": word_a,
        "word_b": word_b,
        "word_id_a": id_a,
        "word_id_b": id_b,
        "similarity": score,
    }


def nearest_neighbours(
    artifact: ModelArtifact,
    word: str,
    top_k: int = 5,
) -> dict[str, Any]:
    word_id, vector = _row(artifact, word)
    if not isinstance(top_k, int) or top_k <= 0:
        raise InputError("top_k must be a positive integer")
    unit = _normalized(vector, word)
    vectors = artifact.input_vectors
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0.0):
        raise ComputationFailureError(
            "vocabulary contains zero or non-finite vectors",
            zero_norm_ids=np.flatnonzero(norms == 0.0).astype(int).tolist(),
        )
    scores = (vectors / norms[:, None]) @ unit
    order = np.argsort(-scores, kind="mergesort")
    words = artifact.vocab.index_to_word
    neighbours: list[dict[str, Any]] = []
    for candidate_id in order.tolist():
        if candidate_id == word_id:
            continue
        neighbours.append(
            {
                "word": words[candidate_id],
                "word_id": int(candidate_id),
                "similarity": float(
                    max(-1.0, min(1.0, scores[candidate_id]))
                ),
            }
        )
        if len(neighbours) >= min(top_k, len(words) - 1):
            break
    return {"word": word, "word_id": word_id, "neighbours": neighbours}


def evaluate_pairs(
    artifact: ModelArtifact,
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate caller-supplied synthetic labelled pairs.

    Each pair is ``{"word_a": str, "word_b": str, "label": 0|1}``. Labels are
    provided by the caller; this method never invents semantic ground truth.
    Agreement is defined as ``(similarity >= threshold) == bool(label)``.
    """
    if not isinstance(pairs, list) or not pairs:
        raise InputError("pairs must be a non-empty list")
    threshold = 0.0
    results = []
    for index, pair in enumerate(pairs):
        try:
            word_a = str(pair["word_a"])
            word_b = str(pair["word_b"])
            label = int(pair["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InputError(
                "each pair needs word_a, word_b and a 0/1 label",
                index=index,
            ) from exc
        if label not in (0, 1):
            raise InputError("label must be 0 or 1", index=index, label=label)
        outcome = cosine_similarity(artifact, word_a, word_b)
        predicted = 1 if outcome["similarity"] >= threshold else 0
        results.append(
            {
                "index": index,
                "word_a": word_a,
                "word_b": word_b,
                "label": label,
                "predicted": predicted,
                "similarity": outcome["similarity"],
                "agrees": predicted == label,
            }
        )
    agreements = sum(1 for item in results if item["agrees"])
    return {
        "pair_count": len(results),
        "threshold": threshold,
        "agreement": agreements / len(results),
        "agreements": agreements,
        "results": results,
        "note": (
            "Agreement with caller-supplied synthetic labels only; "
            "this is not a semantic understanding benchmark."
        ),
    }
