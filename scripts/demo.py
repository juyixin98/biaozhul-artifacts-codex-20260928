#!/usr/bin/env python3
"""Run the bounded NLS solver on fully local synthetic data."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import SolverConfig
from app.core import solve_least_squares
from app.observability import RunLogger
from app.parsing import CompiledModel


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(304)
    x = np.linspace(0.0, 4.0, 30)
    true_parameters = np.array([3.0, 0.7, 0.5])
    y = true_parameters[0] * np.exp(-true_parameters[1] * x) + true_parameters[2]
    y += rng.normal(0.0, 0.01, x.size)

    model = CompiledModel("a*exp(-b*x)+c", ["a", "b", "c"], x.size)
    result = solve_least_squares(
        model=model,
        x=x,
        y=y,
        initial=np.array([0.1, 2.5, 2.0]),
        lower=np.array([0.0, 0.0, -2.0]),
        upper=np.array([10.0, 5.0, 5.0]),
        scales=np.array([1.0, 0.2, 1.0]),
        config=SolverConfig(max_iterations=100, gradient_tol=1.0e-7),
        logger=RunLogger("demo-exp-decay-304"),
    )

    summary = {
        "status": result.status.value,
        "message": result.message,
        "true_parameters": true_parameters.tolist(),
        "parameters": result.parameters,
        "residual_sum_squares": result.residual_sum_squares,
        "iterations": result.iterations,
        "stationarity_type": result.stationarity_type,
        "projected_gradient_norm": result.projected_gradient_norm,
        "condition_number": result.identifiability["condition_number"],
        "initial_derivative_max_abs_error": result.derivative_checks[0]["max_absolute_error"],
        "final_derivative_max_abs_error": result.derivative_checks[-1]["max_absolute_error"],
        "first_reduction_ratio": result.iteration_history[0]["reduction_ratio"],
        "last_reduction_ratio": result.iteration_history[-1]["reduction_ratio"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.status.value == "converged" else 2


if __name__ == "__main__":
    raise SystemExit(main())
