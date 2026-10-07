"""PAV core tests: hand-computed cases plus an independent QP reference.

The reference solves the same constrained least-squares problem with
scipy.optimize (SLSQP), which shares no code with the PAV implementation.
"""
import numpy as np
import pytest
from scipy.optimize import minimize

from calibration_service.app.pav import pav


def qp_reference(y, w):
    """Independent weighted isotonic regression via generic QP solver."""
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    n = len(y)

    def objective(g):
        return np.sum(w * (y - g) ** 2)

    def grad(g):
        return -2 * w * (y - g)

    cons = [
        {"type": "ineq", "fun": lambda g, i=i: g[i + 1] - g[i]}
        for i in range(n - 1)
    ]
    res = minimize(objective, y.copy(), jac=grad, constraints=cons,
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    assert res.success
    return res.x, res.fun


def test_hand_computed_monotone_violation_merges_one_block():
    # y decreases at the end: positions 2,3 violate monotonicity.
    # PAV must pool (0.9, 0.1) into their mean 0.5.
    y = np.array([0.2, 0.4, 0.9, 0.1, 0.8])
    w = np.ones(5)
    res = pav(y, w)
    np.testing.assert_allclose(res.values, [0.2, 0.4, 0.5, 0.5, 0.8], atol=1e-12)
    assert len(res.blocks) == 4


def test_hand_computed_full_collapse_to_weighted_mean():
    # Strictly decreasing sequence collapses into a single block whose
    # value is the weighted mean (1*0.9 + 3*0.2) / 4 = 0.375.
    y = np.array([0.9, 0.2])
    w = np.array([1.0, 3.0])
    res = pav(y, w)
    assert len(res.blocks) == 1
    np.testing.assert_allclose(res.values, [0.375, 0.375], atol=1e-12)
    assert res.blocks[0].weight == pytest.approx(4.0)


def test_weighted_pooling_uses_weights_not_counts():
    # Heavy weight on the low value pulls the pooled level down.
    y = np.array([0.8, 0.2])
    w = np.array([1.0, 7.0])
    res = pav(y, w)
    np.testing.assert_allclose(res.values, [0.275, 0.275], atol=1e-12)


def test_already_monotone_input_is_unchanged():
    y = np.array([0.1, 0.2, 0.2, 0.9])
    w = np.array([1.0, 2.0, 1.0, 1.0])
    res = pav(y, w)
    np.testing.assert_allclose(res.values, y, atol=1e-12)
    # equal adjacent values may stay separate blocks; values are what matter
    assert len(res.blocks) == 4


def test_cascade_merge_path():
    # Merging the last pair must cascade: (0.5, 0.4) -> 0.45, which still
    # violates 0.6, so all three pool to (0.6+0.5+0.4)/3 = 0.5.
    y = np.array([0.1, 0.6, 0.5, 0.4])
    w = np.ones(4)
    res = pav(y, w)
    np.testing.assert_allclose(res.values, [0.1, 0.5, 0.5, 0.5], atol=1e-12)
    assert [(b.start, b.end) for b in res.blocks] == [(0, 1), (1, 4)]


def test_rejects_zero_weight_and_nan():
    with pytest.raises(ValueError, match="strictly positive"):
        pav(np.array([0.1, 0.2]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        pav(np.array([0.1, np.nan]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="empty"):
        pav(np.array([]), np.array([]))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_independent_qp_solution(seed):
    rng = np.random.default_rng(seed)
    n = 12
    y = rng.uniform(0, 1, n)
    w = rng.uniform(0.5, 3.0, n)
    res = pav(y, w)
    g_ref, loss_ref = qp_reference(y, w)
    np.testing.assert_allclose(res.values, g_ref, atol=1e-5)
    # Weighted loss of the PAV solution equals the QP optimum.
    loss_pav = float(np.sum(w * (y - res.values) ** 2))
    assert loss_pav == pytest.approx(loss_ref, abs=1e-6)
