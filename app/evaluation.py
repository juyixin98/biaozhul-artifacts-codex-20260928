"""Synthetic co-occurrence evaluation based on explicit token pairs."""
from __future__ import annotations

import numpy as np

from .errors import ValidationError
from .model import ModelArtifact
from .prediction import token_id


def evaluate_pairs(
    artifact: ModelArtifact, pairs: list[tuple[str, str]], *, include_self: bool = False
) -> dict:
    """Rank each expected partner among all vocabulary rows.

    This is a deterministic synthetic co-occurrence probe.  It is not a
    semantic understanding benchmark.
    """
    if not isinstance(pairs, list) or not pairs:
        raise ValidationError("evaluation pairs must be a non-empty list")

    vectors = artifact.input_matrix
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
        raise ValidationError("cannot evaluate a model containing zero-norm vectors")
    normalized = vectors / norms[:, None]

    results: list[dict] = []
    ranks: list[int] = []
    for position, pair in enumerate(pairs):
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(x, str) for x in pair)
        ):
            raise ValidationError(
                "each evaluation pair must be a [query, expected] string pair",
                {"pair_index": position},
            )
        query, expected = pair
        query_id = token_id(artifact, query)
        expected_id = token_id(artifact, expected)
        scores = normalized @ normalized[query_id]
        target_score = float(scores[expected_id])
        comparison_scores = scores if include_self else np.delete(scores, query_id)
        rank = int(np.sum(comparison_scores > target_score) + 1)
        ranks.append(rank)
        results.append(
            {
                "query": query,
                "expected": expected,
                "rank": rank,
                "score": target_score,
            }
        )

    return {
        "kind": "synthetic_cooccurrence_rank",
        "pair_count": len(results),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "top1_rate": float(np.mean(np.asarray(ranks) == 1)),
        "top3_rate": float(np.mean(np.asarray(ranks) <= 3)),
        "mean_reciprocal_rank": float(np.mean(1.0 / np.asarray(ranks, dtype=float))),
        "results": results,
        "rank_convention": "1_based_among_other_tokens" if not include_self else "including_query",
        "interpretation": (
            "Ranks measure local synthetic co-occurrence recovery. This is not "
            "a semantic understanding benchmark or downstream language test."
        ),
    }
