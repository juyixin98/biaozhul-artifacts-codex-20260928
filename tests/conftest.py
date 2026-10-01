from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def exponential_data():
    rng = np.random.default_rng(304)
    x = np.linspace(0.0, 4.0, 40)
    true = {"a": 3.0, "b": 0.65, "c": 0.4}
    y = true["a"] * np.exp(-true["b"] * x) + true["c"]
    # Tiny reproducible noise keeps the fixture from assuming exact cancellation.
    y = y + rng.normal(scale=1.0e-5, size=x.size)
    return x, y, true


@pytest.fixture
def saturation_data():
    rng = np.random.default_rng(7304)
    x = np.linspace(0.05, 0.8, 35)
    true = {"A": 2.2, "K": 0.18}
    y = true["A"] * x / (true["K"] + x)
    y = y + rng.normal(scale=1.0e-6, size=x.size)
    return x, y, true


@pytest.fixture
def collinear_data():
    rng = np.random.default_rng(9304)
    x = np.linspace(-1.0, 1.0, 24)
    y = 1.25 * x + rng.normal(scale=1.0e-6, size=x.size)
    return x, y
