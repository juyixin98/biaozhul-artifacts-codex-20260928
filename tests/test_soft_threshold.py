"""Soft-threshold operator: hard-coded analytic expectations."""

import numpy as np

from lasso_path.solver import soft_threshold


def test_exact_values():
    assert soft_threshold(3.0, 1.0) == 2.0
    assert soft_threshold(-3.0, 1.0) == -2.0
    assert soft_threshold(0.5, 1.0) == 0.0
    assert soft_threshold(-0.5, 1.0) == 0.0
    # Boundary: exactly at the threshold the result is zero.
    assert soft_threshold(1.0, 1.0) == 0.0
    assert soft_threshold(-1.0, 1.0) == 0.0
    # Zero threshold is the identity.
    assert soft_threshold(-7.25, 0.0) == -7.25


def test_properties_randomized():
    rng = np.random.default_rng(0)
    for _ in range(500):
        z = rng.standard_normal() * 10
        a = abs(rng.standard_normal()) * 2
        s = soft_threshold(z, a)
        assert abs(s) <= abs(z) + 1e-15            # never amplifies
        assert np.sign(s) == np.sign(z) or s == 0.0  # never flips sign
        assert soft_threshold(-z, a) == -s          # odd symmetry
        if abs(z) <= a:
            assert s == 0.0
        else:
            assert s == z - np.sign(z) * a
