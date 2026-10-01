from __future__ import annotations

import logging

import numpy as np
import pytest

from app.config import SolverConfig
from app.observability import RunLogger
from app.parsing import CompiledModel


@pytest.fixture
def run_logger():
    return RunLogger("pytest-run-304", logging.getLogger("pytest.nls"))


@pytest.fixture
def solver_config():
    return SolverConfig(max_iterations=100, gradient_tol=1.0e-7)


@pytest.fixture
def exponential_fixture():
    rng = np.random.default_rng(304)
    x = np.linspace(0.0, 4.0, 30)
    true = np.array([3.0, 0.7, 0.5])
    y = true[0] * np.exp(-true[1] * x) + true[2] + rng.normal(0.0, 0.01, x.size)
    return {
        "x": x,
        "y": y,
        "true": true,
        "model": CompiledModel("a*exp(-b*x)+c", ["a", "b", "c"], x.size),
        "initial": np.array([0.1, 2.5, 2.0]),
        "lower": np.array([0.0, 0.0, -2.0]),
        "upper": np.array([10.0, 5.0, 5.0]),
        "scales": np.array([1.0, 0.2, 1.0]),
    }


@pytest.fixture
def saturation_fixture():
    rng = np.random.default_rng(42)
    x = np.linspace(0.1, 10.0, 40)
    true = np.array([5.0, 2.0])
    y = true[0] * x / (true[1] + x) + rng.normal(0.0, 0.005, x.size)
    return {
        "x": x,
        "y": y,
        "true": true,
        "model": CompiledModel("v*x/(k+x)", ["v", "k"], x.size),
        "initial": np.array([1.0, 8.0]),
        "lower": np.array([0.0, 0.01]),
        "upper": np.array([20.0, 20.0]),
        "scales": np.array([1.0, 0.5]),
    }
