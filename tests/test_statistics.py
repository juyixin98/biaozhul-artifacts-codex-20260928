from __future__ import annotations

import numpy as np
import pytest

from lag_correlation.kernel.statistics import linear_detrend, pearson_from_residuals


def test_linear_detrend_removes_fitted_line() -> None:
    times = np.arange(6.0)
    values = 3.0 + 2.0 * times

    residual = linear_detrend(times, values)

    assert np.max(np.abs(residual)) < 1e-12


def test_pearson_for_known_residuals() -> None:
    times = np.arange(6.0)
    left = np.array([1.0, 3.0, 2.0, 8.0, 4.0, 7.0])
    right = 2.0 * left + 5.0 * times - 1.0

    correlation = pearson_from_residuals(
        linear_detrend(times, left),
        linear_detrend(times, right),
    )

    assert correlation == pytest.approx(1.0, abs=1e-12)


def test_constant_residuals_are_rejected() -> None:
    correlation = pearson_from_residuals(
        np.zeros(6),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )

    assert correlation is None
