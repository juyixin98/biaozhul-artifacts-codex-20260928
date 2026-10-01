"""Generate deterministic synthetic fixtures and independent reference answers.

The project's decomposition kernel is intentionally not imported here.
Reference components come from statsmodels' mature STL implementation; input
series come from explicit trend, seasonal, noise, and outlier formulas.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from statsmodels.tsa.seasonal import STL


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "fixtures"


def payload_hash(values: list[float]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def centered_statsmodels(values: np.ndarray) -> dict[str, list[float]]:
    fit = STL(
        values,
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
    ).fit(inner_iter=2, outer_iter=5)
    seasonal_shift = float(np.mean(fit.seasonal))
    seasonal = fit.seasonal - seasonal_shift
    trend = fit.trend + seasonal_shift
    residual = values - trend - seasonal
    return {
        "trend": np.asarray(trend, dtype=np.float64).tolist(),
        "seasonal": np.asarray(seasonal, dtype=np.float64).tolist(),
        "residual": np.asarray(residual, dtype=np.float64).tolist(),
        "weights": np.asarray(fit.weights, dtype=np.float64).tolist(),
    }


def generate_primary_fixture() -> dict:
    rng = np.random.default_rng(20260930)
    n = 70
    t = np.arange(n, dtype=np.float64)
    true_trend = 0.08 * t + 0.002 * t**2
    true_seasonal = 3.0 * np.sin(2 * np.pi * t / 7.0) + 1.5 * np.cos(
        4 * np.pi * t / 7.0
    )
    noise = rng.normal(0.0, 0.22, size=n)
    outlier_indices = [9, 31, 57]
    outlier_shocks = np.array([10.0, -8.0, 7.0])
    values = true_trend + true_seasonal + noise
    values[np.asarray(outlier_indices)] += outlier_shocks
    reference = centered_statsmodels(values)
    return {
        "fixture_version": "1.0.0",
        "name": "linear_trend_sinusoidal_season_three_outliers",
        "description": (
            "70 samples = linear/quadratic trend + period-7 seasonal signal "
            "+ deterministic-seed Gaussian noise + three additive outliers."
        ),
        "period": 7,
        "n": n,
        "seed": 20260930,
        "noise_distribution": "numpy.default_rng.normal(loc=0, scale=0.22)",
        "outlier_indices": outlier_indices,
        "outlier_shocks": outlier_shocks.tolist(),
        "formula": {
            "trend": "0.08*t + 0.002*t^2",
            "seasonal": "3*sin(2*pi*t/7) + 1.5*cos(4*pi*t/7)",
            "observed": "trend + seasonal + noise + additive_outliers",
        },
        "values": values.tolist(),
        "true_components": {
            "trend": true_trend.tolist(),
            "seasonal": true_seasonal.tolist(),
            "noise": noise.tolist(),
        },
        "statsmodels_reference": {
            "package": "statsmodels",
            "version": "0.15.0",
            "implementation": "statsmodels.tsa.seasonal.STL",
            "arguments": {
                "period": 7,
                "seasonal": 7,
                "trend": 15,
                "low_pass": 9,
                "seasonal_deg": 1,
                "trend_deg": 1,
                "low_pass_deg": 1,
                "robust": True,
                "inner_iter": 2,
                "outer_iter": 5,
            },
            "centering": "reference seasonal centered independently for identifiability",
            **reference,
        },
        "hashes": {
            "input_sha256": payload_hash(values.tolist()),
            "reference_trend_sha256": payload_hash(reference["trend"]),
            "reference_seasonal_sha256": payload_hash(reference["seasonal"]),
            "reference_weights_sha256": payload_hash(reference["weights"]),
        },
    }


def generate_boundary_fixture() -> dict:
    rng = np.random.default_rng(314159)
    n = 14
    t = np.arange(n, dtype=np.float64)
    seasonal = np.array([1.0, -0.5, 0.75, 2.0, -2.0, 0.25, -1.5])
    values = 0.1 * t + np.tile(seasonal, 2) + rng.normal(0, 0.03, n)
    reference = centered_statsmodels(values)
    return {
        "fixture_version": "1.0.0",
        "name": "minimum_two_period_boundary_fixture",
        "description": "Exactly two periods; exercises STL seasonal extensions and endpoint windows.",
        "period": 7,
        "n": n,
        "seed": 314159,
        "values": values.tolist(),
        "statsmodels_reference": {
            "package": "statsmodels",
            "version": "0.15.0",
            "implementation": "statsmodels.tsa.seasonal.STL",
            **reference,
        },
        "hashes": {
            "input_sha256": payload_hash(values.tolist()),
            "reference_trend_sha256": payload_hash(reference["trend"]),
            "reference_seasonal_sha256": payload_hash(reference["seasonal"]),
        },
    }


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = [
        ("synthetic_stl_70.json", generate_primary_fixture()),
        ("minimum_periods_14.json", generate_boundary_fixture()),
    ]
    for filename, fixture in fixtures:
        path = FIXTURE_DIR / filename
        path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", "utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

