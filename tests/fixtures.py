"""Deterministic synthetic fixtures and independent reference calculations.

These functions intentionally do not import the application's estimators.
Tests use them as independently written expected-value sources.
"""

from __future__ import annotations

import numpy as np


def normalish_observations(n: int = 50, scale: float = 1.0) -> np.ndarray:
    """Return a fixed bounded pseudo-normal-like sequence without an RNG dependency."""

    values = np.array(
        [
            0.15,
            -0.42,
            0.88,
            -1.15,
            1.46,
            0.31,
            -0.66,
            0.72,
            -0.93,
            0.05,
            1.31,
            -1.48,
            0.58,
            -0.24,
            0.99,
            -0.78,
            0.36,
            -1.30,
            1.07,
            -0.08,
        ],
        dtype=float,
    )
    repeated = np.array([values[i % len(values)] for i in range(n)], dtype=float)
    return repeated * scale


def clean_rescaled_dataset(n: int = 50) -> np.ndarray:
    base = normalish_observations(n)
    data = np.column_stack(
        [
            base,
            2.0 * base + 0.35 * normalish_observations(n + 3)[3:],
            4.0 * base - 0.55 * normalish_observations(n + 7)[7:],
            100.0 * base + 30.0 * normalish_observations(n + 11)[11:],
        ]
    )
    return data


def strong_outlier_dataset(n: int = 50) -> tuple[np.ndarray, list[int]]:
    data = clean_rescaled_dataset(n)
    outlier_indices = [5, 18, 37]
    data[outlier_indices[0]] += np.array([12.0, -18.0, 28.0, 1400.0])
    data[outlier_indices[1]] += np.array([-10.0, 20.0, -32.0, -1200.0])
    data[outlier_indices[2]] += np.array([14.0, 16.0, 30.0, 1500.0])
    return data, outlier_indices


def collinear_dataset(n: int = 40) -> np.ndarray:
    base = normalish_observations(n)
    second = normalish_observations(n + 5)[5:]
    return np.column_stack([base, 2.0 * base + 1.0, 3.0 * base - 0.5, 50.0 * second])


def high_dimensional_dataset(n: int = 8, p: int = 12, seed: int = 319) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p))


def missing_dataset() -> tuple[np.ndarray, list[list[float | None]], list[float]]:
    data = clean_rescaled_dataset(20).copy()
    rows: list[list[float | None]] = data.tolist()
    rows[1][0] = None
    rows[4][3] = None
    medians = [
        float(np.median(np.delete(data[:, column], row)))
        for row, column in [(1, 0), (4, 3)]
    ]
    data[1, 0] = medians[0]
    data[4, 3] = medians[1]
    return data, rows, medians


def independent_empirical_covariance(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = data.mean(axis=0)
    covariance = np.cov(data, rowvar=False, bias=True)
    return location, covariance


def independent_ledoit_wolf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    location = data.mean(axis=0)
    centered = data - location
    n, p = data.shape
    sample_cov = np.zeros((p, p), dtype=float)
    for row in centered:
        sample_cov += np.outer(row, row)
    sample_cov /= n
    target_variance = float(np.trace(sample_cov) / p)
    target = target_variance * np.eye(p)
    delta = float(np.sum((sample_cov - target) ** 2) / p)
    beta_sum = 0.0
    for row in centered:
        outer = np.outer(row, row)
        beta_sum += np.sum((outer - sample_cov) ** 2)
    beta = min(beta_sum / (n * n * p), delta)
    shrinkage = beta / delta if delta > 0 else 0.0
    return location, (1.0 - shrinkage) * sample_cov + shrinkage * target, shrinkage


def independent_mad(values: np.ndarray) -> np.ndarray:
    center = np.median(values, axis=0)
    return np.median(np.abs(values - center), axis=0) / 0.6744897501960817
