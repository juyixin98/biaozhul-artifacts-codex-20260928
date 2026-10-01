"""Independent synthetic fixtures and reference calculations.

These helpers deliberately do not import the production statistical kernel.
Expected correlations are calculated directly from the ordinary least-squares
definition and the sample Pearson formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


VALUES = (1.0, 5.0, 2.0, 9.0, 4.0, 8.0, 3.0, 7.0, 2.0, 6.0)
KNOWN_LAG = 2.0


@dataclass(frozen=True)
class Scenario:
    name: str
    x_name: str
    y_name: str
    x_times: list[float]
    x_values: list[float]
    y_times: list[float]
    y_values: list[float]
    expected_status: str
    expected_failure_category: str | None
    expected_peak_lag: float | None
    expected_peak_correlation: float | None
    expected_best_n_pairs: int
    expected_dense_used: int
    expected_dense_unmatched: int
    expected_anchor_side: str
    expected_first_pair: dict[str, Any]
    lag_grid: tuple[float, float, float]
    tolerance: float


def scenarios() -> list[Scenario]:
    dense = _different_density_scenario()
    uniform = _uniform_scenario()
    gap = _long_gap_scenario()
    constant = _constant_scenario()
    no_overlap = _no_overlap_scenario()
    return [dense, uniform, gap, constant, no_overlap]


def reference_detrended_pearson(
    times: list[float], left: list[float], right: list[float]
) -> float | None:
    time_array = np.asarray(times, dtype=np.float64)
    arrays = [np.asarray(values, dtype=np.float64) for values in (left, right)]
    residuals = []
    for values in arrays:
        slope, intercept = np.polyfit(time_array, values, 1)
        residual = values - (slope * time_array + intercept)
        if np.std(residual, ddof=1) == 0.0:
            return None
        residuals.append((residual - np.mean(residual)) / np.std(residual, ddof=1))
    return float(np.clip(np.dot(residuals[0], residuals[1]) / (len(time_array) - 1), -1.0, 1.0))


def dense_expected_pair() -> dict[str, Any]:
    return {
        "anchor_side": "x",
        "anchor_time": 0.0,
        "paired_time": -2.0,
        "anchor_index": 0,
        "paired_indices": [0, 1, 2],
        "time_delta": 2.0,
        "lag": 2.0,
    }


def _different_density_scenario() -> Scenario:
    x_times = [float(index * 10) for index in range(10)]
    y_times: list[float] = []
    y_values: list[float] = []
    for anchor_time, value in zip(x_times, VALUES):
        for offset, value_offset in ((-0.2, -0.2), (0.0, 0.0), (0.2, 0.2)):
            y_times.append(anchor_time - KNOWN_LAG + offset)
            y_values.append(value + value_offset)

    return Scenario(
        name="known_offset_different_density",
        x_name="sparse_x",
        y_name="dense_y",
        x_times=x_times,
        x_values=list(VALUES),
        y_times=y_times,
        y_values=y_values,
        expected_status="ok",
        expected_failure_category=None,
        expected_peak_lag=KNOWN_LAG,
        expected_peak_correlation=1.0,
        expected_best_n_pairs=10,
        expected_dense_used=30,
        expected_dense_unmatched=0,
        expected_anchor_side="x",
        expected_first_pair=dense_expected_pair(),
        lag_grid=(-5.0, 5.0, 1.0),
        tolerance=0.25,
    )


def _uniform_scenario() -> Scenario:
    y_values = list(VALUES[2:]) + [5.0, 1.0]
    return Scenario(
        name="uniform_sampling_special_case",
        x_name="uniform_x",
        y_name="uniform_shifted_y",
        x_times=[float(index) for index in range(10)],
        x_values=list(VALUES),
        y_times=[float(index) for index in range(10)],
        y_values=y_values,
        expected_status="ok",
        expected_failure_category=None,
        expected_peak_lag=KNOWN_LAG,
        expected_peak_correlation=1.0,
        expected_best_n_pairs=8,
        expected_dense_used=8,
        expected_dense_unmatched=2,
        expected_anchor_side="x",
        expected_first_pair={
            "anchor_side": "x",
            "anchor_time": 2.0,
            "paired_time": 0.0,
            "anchor_index": 2,
            "paired_indices": [0],
            "time_delta": 2.0,
            "lag": 2.0,
        },
        lag_grid=(-5.0, 5.0, 1.0),
        tolerance=0.1,
    )


def _long_gap_scenario() -> Scenario:
    x_times = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 60.0, 61.0, 62.0, 63.0]
    y_times = [time - KNOWN_LAG for time in x_times]
    return Scenario(
        name="known_offset_with_long_gap",
        x_name="gap_x",
        y_name="gap_y",
        x_times=x_times,
        x_values=list(VALUES),
        y_times=y_times,
        y_values=list(VALUES),
        expected_status="ok",
        expected_failure_category=None,
        expected_peak_lag=KNOWN_LAG,
        expected_peak_correlation=1.0,
        expected_best_n_pairs=10,
        expected_dense_used=10,
        expected_dense_unmatched=0,
        expected_anchor_side="x",
        expected_first_pair={
            "anchor_side": "x",
            "anchor_time": 10.0,
            "paired_time": 8.0,
            "anchor_index": 0,
            "paired_indices": [0],
            "time_delta": 2.0,
            "lag": 2.0,
        },
        lag_grid=(-5.0, 5.0, 1.0),
        tolerance=0.1,
    )


def _constant_scenario() -> Scenario:
    times = [float(index) for index in range(10)]
    return Scenario(
        name="constant_after_detrend",
        x_name="constant_x",
        y_name="constant_y",
        x_times=times,
        x_values=[1.0] * 10,
        y_times=times,
        y_values=[7.0] * 10,
        expected_status="unknown",
        expected_failure_category="INSUFFICIENT_VARIANCE",
        expected_peak_lag=None,
        expected_peak_correlation=None,
        expected_best_n_pairs=10,
        expected_dense_used=10,
        expected_dense_unmatched=0,
        expected_anchor_side="x",
        expected_first_pair={
            "anchor_side": "x",
            "anchor_time": 0.0,
            "paired_time": 0.0,
            "anchor_index": 0,
            "paired_indices": [0],
            "time_delta": 0.0,
            "lag": 0.0,
        },
        lag_grid=(-5.0, 5.0, 1.0),
        tolerance=0.1,
    )


def _no_overlap_scenario() -> Scenario:
    return Scenario(
        name="no_effective_overlap",
        x_name="early_x",
        y_name="late_y",
        x_times=[float(index) for index in range(10)],
        x_values=list(VALUES),
        y_times=[100.0 + float(index) for index in range(10)],
        y_values=list(reversed(VALUES)),
        expected_status="unknown",
        expected_failure_category="INSUFFICIENT_OVERLAP",
        expected_peak_lag=None,
        expected_peak_correlation=None,
        expected_best_n_pairs=0,
        expected_dense_used=0,
        expected_dense_unmatched=10,
        expected_anchor_side="x",
        expected_first_pair={},
        lag_grid=(-5.0, 5.0, 1.0),
        tolerance=0.1,
    )
