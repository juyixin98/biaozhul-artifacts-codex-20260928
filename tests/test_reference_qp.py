from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, minimize

from app.kernel import aggregate_equal_scores, fit_pav
from app.validation import CleanPoint


def weighted_qp_reference(points: list[CleanPoint]) -> tuple[np.ndarray, np.ndarray]:
    grouped = aggregate_equal_scores(points)
    scores = np.array([point.score for point in grouped], dtype=float)
    weights = np.array([point.weight for point in grouped], dtype=float)
    targets = np.array([point.target for point in grouped], dtype=float)

    def objective(probabilities: np.ndarray) -> float:
        return float(np.sum(weights * (probabilities - targets) ** 2))

    if len(scores) == 1:
        constraint_matrix = np.zeros((0, 1))
    else:
        # Each row is p_i - p_(i+1) <= 0, equivalent to p_i <= p_(i+1).
        constraint_matrix = np.eye(len(scores) - 1, len(scores)) - np.eye(len(scores) - 1, len(scores), k=1)
    constraints = LinearConstraint(constraint_matrix, lb=-np.inf, ub=0.0)
    result = minimize(
        objective,
        x0=targets,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(scores),
        constraints=[constraints],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert result.success, result.message
    return scores, result.x


def test_pav_matches_independent_constrained_qp_on_small_cases():
    cases = [
        [
            CleanPoint(score=0.0, target=1, weight=1.0, record_id=None),
            CleanPoint(score=1.0, target=0, weight=2.0, record_id=None),
            CleanPoint(score=2.0, target=1, weight=1.5, record_id=None),
            CleanPoint(score=3.0, target=0, weight=0.5, record_id=None),
            CleanPoint(score=4.0, target=1, weight=3.0, record_id=None),
        ],
        [
            CleanPoint(-2.0, 0, 4.0, None),
            CleanPoint(-2.0, 1, 1.0, None),
            CleanPoint(0.0, 1, 2.0, None),
            CleanPoint(0.0, 0, 3.0, None),
            CleanPoint(5.0, 1, 1.0, None),
            CleanPoint(7.0, 0, 5.0, None),
        ],
        [
            CleanPoint(10.0, 1, 0.1, None),
            CleanPoint(11.0, 1, 0.1, None),
            CleanPoint(12.0, 0, 10.0, None),
            CleanPoint(13.0, 1, 0.2, None),
        ],
    ]

    for case in cases:
        scores, qp_probabilities = weighted_qp_reference(case)
        pav = fit_pav(case)
        probability_by_score = {
            score: probability
            for block, probability in zip(pav.blocks, pav.probabilities, strict=True)
            for score_index in block.score_indexes
            for score in [grouped_score(case, score_index)]
        }
        pav_expanded = np.array([probability_by_score[float(score)] for score in scores])
        np.testing.assert_allclose(np.asarray([block.min_score for block in pav.blocks]), pav.scores)
        np.testing.assert_allclose(pav_expanded, qp_probabilities, atol=1e-8)


def test_pav_weighted_loss_matches_reference_objective():
    points = [
        CleanPoint(0.0, 1, 2.0, None),
        CleanPoint(1.0, 0, 3.0, None),
        CleanPoint(2.0, 1, 5.0, None),
        CleanPoint(3.0, 0, 1.0, None),
    ]
    pav = fit_pav(points)
    grouped = aggregate_equal_scores(points)

    by_index = {index: point for index, point in enumerate(grouped)}
    expected = 0.0
    for block, probability in zip(pav.blocks, pav.probabilities, strict=True):
        expected += sum(
            by_index[index].weight * (probability - by_index[index].target) ** 2
            for index in block.score_indexes
        )
    assert pav.weighted_sse == pytest.approx(expected)


def grouped_score(points, score_index: int) -> float:
    return aggregate_equal_scores(points)[score_index].score
