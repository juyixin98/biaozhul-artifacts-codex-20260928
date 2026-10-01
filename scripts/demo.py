#!/usr/bin/env python3
"""Run all local synthetic trust-region demonstrations without external services."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from app.diagnostics import RunLogger
from app.solver import SolveOptions, solve_nonlinear_least_squares


LOG_PATH = ROOT / "logs" / "demo.jsonl"


def report(title: str, result, truth: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    print(f"success={result.success} status={result.status.value}")
    print(f"reason={result.reason}")
    for name, expected in truth.items():
        got = result.parameters.get(name, float("nan"))
        print(f"  {name:5s} estimated={got: .8f} reference={expected: .8f} error={got - expected: .3e}")
    residuals = np.asarray(result.residuals, dtype=float)
    if residuals.size:
        print(f"  residual RMS={np.sqrt(np.mean(residuals**2)): .6e}")
    print(f"  iterations={result.iterations} accepted={result.accepted_iterations}")
    if result.diagnostics.finite_difference_max_error is not None:
        print(f"  AD/finite-difference max relative error={result.diagnostics.finite_difference_max_error: .3e}")
    if result.diagnostics.condition_number is not None:
        print(f"  scaled Jacobian condition number={result.diagnostics.condition_number: .3e}")
    for item in result.diagnostics.iterations[-3:]:
        print(
            "  iter={iteration:02d} accepted={accepted} cost={cost:.6e} "
            "predicted={predicted_reduction:.6e} actual={actual_reduction:.6e} "
            "rho={reduction_ratio:.6f} radius={trust_radius:.6g}".format(**vars(item))
        )


def main() -> None:
    rng = np.random.default_rng(20260930)

    x_exp = np.linspace(0.0, 2.0, 18)
    exp_truth = {"amp": 2.0, "rate": 1.5, "floor": 0.3}
    y_exp = (
        exp_truth["amp"] * np.exp(-exp_truth["rate"] * x_exp)
        + exp_truth["floor"]
        + rng.normal(0.0, 4e-4, x_exp.size)
    )
    exp_result = solve_nonlinear_least_squares(
        "amp*exp(-rate*x)+floor",
        ["amp", "rate", "floor"],
        x_exp,
        y_exp,
        initial={"amp": 0.05, "rate": 0.01, "floor": 1.6},
        lower={"amp": 0.0, "rate": 0.0, "floor": 0.0},
        upper={"amp": 8.0, "rate": 8.0, "floor": 8.0},
        scales={"amp": 2.0, "rate": 1.5, "floor": 1.0},
        options=SolveOptions(max_iterations=100),
        logger=RunLogger(LOG_PATH, run_id="demo-exponential"),
    )
    report("Synthetic exponential decay; bad initial value", exp_result, exp_truth)

    x_sat = np.linspace(0.0, 4.0, 28)
    sat_truth = {"a": 1.2, "k": 0.8, "b": 0.2}
    approach = 1.0 - np.exp(-sat_truth["k"] * x_sat)
    y_sat = (
        sat_truth["a"] * approach / (1.0 + sat_truth["b"] * approach)
        + rng.normal(0.0, 3e-4, x_sat.size)
    )
    sat_result = solve_nonlinear_least_squares(
        "a*(1-exp(-k*x))/(1+b*(1-exp(-k*x)))",
        ["a", "k", "b"],
        x_sat,
        y_sat,
        initial={"a": 0.08, "k": 0.03, "b": 1.4},
        lower={"a": 0.0, "k": 0.0, "b": -2.0},
        upper={"a": 5.0, "k": 5.0, "b": 5.0},
        scales={"a": 1.0, "k": 0.8, "b": 0.5},
        options=SolveOptions(max_iterations=100),
        logger=RunLogger(LOG_PATH, run_id="demo-saturation"),
    )
    report("Synthetic approach-to-saturation model", sat_result, sat_truth)

    x_bad = np.linspace(0.0, 2.0, 10)
    illegal = solve_nonlinear_least_squares(
        "amp*sqrt(x-offset)",
        ["amp", "offset"],
        x_bad,
        np.zeros_like(x_bad),
        initial={"amp": 1.0, "offset": 2.0},
        lower={"amp": 0.0, "offset": 0.0},
        upper={"amp": 3.0, "offset": 3.0},
        options=SolveOptions(derivative_check=False),
        logger=RunLogger(LOG_PATH, run_id="demo-illegal-domain"),
    )
    print("\n=== Illegal-domain diagnosis ===")
    print(f"success={illegal.success} status={illegal.status.value} reason={illegal.reason}")

    redundant = solve_nonlinear_least_squares(
        "a+b+0*x",
        ["a", "b"],
        x_bad,
        np.zeros_like(x_bad),
        initial={"a": 0.2, "b": 0.4},
        lower={"a": -1.0, "b": -1.0},
        upper={"a": 1.0, "b": 1.0},
        options=SolveOptions(derivative_check=False),
        logger=RunLogger(LOG_PATH, run_id="demo-unidentifiable"),
    )
    print("\n=== Unidentifiable diagnosis ===")
    print(
        f"success={redundant.success} status={redundant.status.value} "
        f"rank={redundant.diagnostics.rank} reason={redundant.reason}"
    )
    print(f"\nJSONL diagnostics written to {LOG_PATH}")


if __name__ == "__main__":
    main()
