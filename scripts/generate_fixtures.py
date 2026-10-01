#!/usr/bin/env python3
"""Generate deterministic local STL fixtures and independent reference answers."""

from __future__ import annotations

import json
import pathlib

import numpy as np
from statsmodels.tsa.seasonal import STL


FIXTURE_PATH = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def generate_robust_fixture() -> dict:
    rng = np.random.default_rng(20260930)
    n = 140
    index = np.arange(n, dtype=np.float64)
    trend = 2.0 + 0.08 * index
    seasonal = 0.9 * np.sin(2.0 * np.pi * index / 7.0)
    noise = rng.normal(0.0, 0.06, size=n)
    observed = trend + seasonal + noise

    outliers = {
        "24": 7.5,
        "71": -6.25,
        "117": 6.8,
    }
    for position, shift in outliers.items():
        observed[int(position)] += shift

    reference = STL(
        observed,
        period=7,
        seasonal=7,
        trend=15,
        low_pass=9,
        seasonal_deg=1,
        trend_deg=1,
        low_pass_deg=1,
        seasonal_jump=1,
        trend_jump=1,
        low_pass_jump=1,
        robust=True,
    ).fit(inner_iter=2, outer_iter=15)

    # The implementation under test adds exact sample-mean centering after the
    # canonical STL loop. Apply that independent identity to the reference too.
    reference_seasonal = np.asarray(reference.seasonal)
    reference_seasonal_centered = reference_seasonal - np.mean(reference_seasonal)
    reference_trend_centered = np.asarray(reference.trend) + np.mean(
        reference_seasonal
    )
    reference_residual_centered = (
        observed - reference_trend_centered - reference_seasonal_centered
    )

    return {
        "description": "Linear trend, deterministic sine period 7, Gaussian noise, outliers.",
        "generator": "scripts/generate_fixtures.py",
        "reference_implementation": "statsmodels.tsa.seasonal.STL 0.15.0",
        "random_seed": 20260930,
        "outlier_positions": [int(k) for k in outliers],
        "outlier_shifts": [float(v) for v in outliers.values()],
        "known_components": {
            "trend_formula": "2.0 + 0.08 * index",
            "seasonal_formula": "0.9 * sin(2*pi*index/7)",
            "noise_mean": 0.0,
            "noise_standard_deviation": 0.06,
        },
        "config": {
            "period": 7,
            "seasonal_window": 7,
            "trend_window": 15,
            "low_pass_window": 9,
            "seasonal_degree": 1,
            "trend_degree": 1,
            "low_pass_degree": 1,
            "inner_iterations": 2,
            "outer_iterations": 15,
        },
        "observed": [float(x) for x in observed],
        "known_trend": [float(x) for x in trend],
        "known_seasonal": [float(x) for x in seasonal],
        "reference": {
            "trend": [float(x) for x in reference_trend_centered],
            "seasonal": [float(x) for x in reference_seasonal_centered],
            "residual": [float(x) for x in reference_residual_centered],
            "robust_weights": [float(x) for x in reference.weights],
        },
    }


def main() -> None:
    FIXTURE_PATH.mkdir(parents=True, exist_ok=True)
    fixture = generate_robust_fixture()
    target = FIXTURE_PATH / "robust_stl.json"
    target.write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n", "utf-8"
    )
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
