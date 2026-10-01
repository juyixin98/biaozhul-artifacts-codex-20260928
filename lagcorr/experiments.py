"""Deterministic synthetic experiments for reproducibility checks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import pearsonr

from .kernel import AnalysisConfig, AnalysisOutcome, SeriesInput, analyze_lags


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    a: SeriesInput
    b: SeriesInput
    lags: list[float]
    tolerance: float
    expected_peak_lag: float | None
    expected_peak_correlation: float | None
    expected_statuses: dict[float, str]
    expected_pairs: dict[float, list[tuple[int, int]]]


def independent_one_to_one_pairs(
    a_times: list[float],
    b_times: list[float],
    lag: float,
    tolerance: float,
) -> list[tuple[int, int]]:
    """Small, intentionally independent brute-force oracle for test data."""

    edges = []
    for i, a_time in enumerate(a_times):
        for j, b_time in enumerate(b_times):
            residual = b_time - lag - a_time
            if abs(residual) <= tolerance:
                edges.append((abs(residual), i, j))

    selected = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for _, i, j in sorted(edges):
        if i not in used_a and j not in used_b:
            selected.append((i, j))
            used_a.add(i)
            used_b.add(j)
    return selected


def build_cases() -> list[ExperimentCase]:
    # Uniform sampled special case. The reference Pearson value comes directly
    # from SciPy on independently listed exact pairs.
    uniform_a_times = [float(i) for i in range(20)]
    uniform_b_times = [i + 3.0 for i in range(20)]
    uniform_a_values = [
        0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0,
        4.0, 6.0, 5.0, 5.0, 6.0, 4.0, 7.0, 3.0, 8.0, 2.0,
    ]
    uniform_b_values = [
        value + 0.25 * math.sin(index * 1.7)
        for index, value in enumerate(uniform_a_values)
    ]
    uniform_expected_r = float(
        pearsonr(uniform_a_values, uniform_b_values).statistic
    )

    # Dense B has explicit decoy points 0.5 away. True points are residual 0.
    dense_a_times = [2.0 * i for i in range(20)]
    dense_a_values = [math.sin(i / 3.0) for i in range(20)]
    dense_b_times: list[float] = []
    dense_b_values: list[float] = []
    for i, value in enumerate(dense_a_values):
        dense_b_times.append(2.0 * i + 0.5)
        dense_b_values.append(999.0 if i % 2 == 0 else -999.0)
        dense_b_times.append(2.0 * i + 1.0)
        dense_b_values.append(value)

    gap_a_times = [float(i) for i in range(8)] + [100.0 + i for i in range(8)]
    gap_a_values = [math.cos(i / 4.0) for i in range(16)]
    gap_b_times = [time + 0.25 for time in gap_a_times]
    gap_b_values = list(gap_a_values)

    constant_a_times = [10.0 * i for i in range(16)]
    constant_b_times = [time + 1.0 for time in constant_a_times]

    irregular_a_times = [10.0 * i for i in range(20)]
    irregular_b_times = [time + 2.0 for time in irregular_a_times]
    irregular_a_values = [i - 9.5 for i in range(20)]
    irregular_b_values = [
        -0.4 * a_value + math.sin(a_value)
        for a_value in irregular_a_values
    ]
    irregular_expected_r = float(
        pearsonr(irregular_a_values, irregular_b_values).statistic
    )

    return [
        ExperimentCase(
            name="uniform_known_shift",
            a=SeriesInput(uniform_a_times, uniform_a_values, "A"),
            b=SeriesInput(uniform_b_times, uniform_b_values, "B"),
            lags=[3.0, 19.0],
            tolerance=0.4,
            expected_peak_lag=3.0,
            expected_peak_correlation=uniform_expected_r,
            expected_statuses={3.0: "accepted", 19.0: "unknown"},
            expected_pairs={
                3.0: independent_one_to_one_pairs(
                    uniform_a_times, uniform_b_times, 3.0, 0.4
                ),
                19.0: independent_one_to_one_pairs(
                    uniform_a_times, uniform_b_times, 19.0, 0.4
                ),
            },
        ),
        ExperimentCase(
            name="dense_side_one_to_one",
            a=SeriesInput(dense_a_times, dense_a_values, "A"),
            b=SeriesInput(dense_b_times, dense_b_values, "B"),
            lags=[1.0],
            tolerance=0.6,
            expected_peak_lag=1.0,
            expected_peak_correlation=1.0,
            expected_statuses={1.0: "accepted"},
            expected_pairs={
                1.0: independent_one_to_one_pairs(
                    dense_a_times, dense_b_times, 1.0, 0.6
                )
            },
        ),
        ExperimentCase(
            name="long_gap",
            a=SeriesInput(gap_a_times, gap_a_values, "A"),
            b=SeriesInput(gap_b_times, gap_b_values, "B"),
            lags=[0.25],
            tolerance=0.3,
            expected_peak_lag=0.25,
            expected_peak_correlation=1.0,
            expected_statuses={0.25: "accepted"},
            expected_pairs={
                0.25: independent_one_to_one_pairs(
                    gap_a_times, gap_b_times, 0.25, 0.3
                )
            },
        ),
        ExperimentCase(
            name="constant_sequence",
            a=SeriesInput(constant_a_times, [1.0] * 16, "A"),
            b=SeriesInput(constant_b_times, [2.0] * 16, "B"),
            lags=[1.0],
            tolerance=0.5,
            expected_peak_lag=None,
            expected_peak_correlation=None,
            expected_statuses={1.0: "rejected"},
            expected_pairs={
                1.0: independent_one_to_one_pairs(
                    constant_a_times, constant_b_times, 1.0, 0.5
                )
            },
        ),
        ExperimentCase(
            name="insufficient_overlap",
            a=SeriesInput(uniform_a_times, uniform_a_values, "A"),
            b=SeriesInput(uniform_b_times, uniform_b_values, "B"),
            lags=[80.0],
            tolerance=0.4,
            expected_peak_lag=None,
            expected_peak_correlation=None,
            expected_statuses={80.0: "unknown"},
            expected_pairs={80.0: []},
        ),
        ExperimentCase(
            name="irregular_nonperfect_uniform_comparison",
            a=SeriesInput(irregular_a_times, irregular_a_values, "A"),
            b=SeriesInput(irregular_b_times, irregular_b_values, "B"),
            lags=[2.0],
            tolerance=3.0,
            expected_peak_lag=2.0,
            expected_peak_correlation=irregular_expected_r,
            expected_statuses={2.0: "accepted"},
            expected_pairs={
                2.0: independent_one_to_one_pairs(
                    irregular_a_times, irregular_b_times, 2.0, 3.0
                )
            },
        ),
    ]


def assert_case(case: ExperimentCase) -> None:
    outcome = analyze_lags(
        case.a,
        case.b,
        case.lags,
        case.tolerance,
        AnalysisConfig(),
    )
    by_lag = {result.lag: result for result in outcome.results}

    for lag, expected_status in case.expected_statuses.items():
        result = by_lag[lag]
        assert result.status == expected_status, (
            f"{case.name} lag={lag}: expected {expected_status}, got {result.status}"
        )
        actual_pairs = [(pair.a_index, pair.b_index) for pair in result.pairs]
        expected_pairs = case.expected_pairs[lag]
        assert actual_pairs == expected_pairs, (
            f"{case.name} lag={lag}: pair mismatch "
            f"expected={expected_pairs} actual={actual_pairs}"
        )

    assert outcome.peak_lag == case.expected_peak_lag
    if case.expected_peak_correlation is None:
        assert outcome.peak_correlation is None
    else:
        assert outcome.peak_correlation is not None
        assert abs(outcome.peak_correlation - case.expected_peak_correlation) < 1e-12

    # Pair evidence must be endpoint-unique whenever pairs are returned.
    for result in outcome.results:
        a_indices = [pair.a_index for pair in result.pairs]
        b_indices = [pair.b_index for pair in result.pairs]
        assert len(a_indices) == len(set(a_indices))
        assert len(b_indices) == len(set(b_indices))
        assert all(abs(pair.residual) <= case.tolerance for pair in result.pairs)


def run_experiments() -> list[tuple[str, AnalysisOutcome]]:
    output = []
    for case in build_cases():
        assert_case(case)
        outcome = analyze_lags(
            case.a, case.b, case.lags, case.tolerance, AnalysisConfig()
        )
        output.append((case.name, outcome))
    return output


def main() -> None:
    for case in build_cases():
        assert_case(case)
        outcome = analyze_lags(
            case.a, case.b, case.lags, case.tolerance, AnalysisConfig()
        )
        statuses = ", ".join(
            f"lag={result.lag}:{result.status}"
            f"{'' if result.correlation is None else ':r=' + format(result.correlation, '.12f')}"
            f":pairs={result.pair_count}"
            for result in outcome.results
        )
        print(f"PASS {case.name} [{statuses}]")
    print("All reproducibility experiments passed.")


if __name__ == "__main__":
    main()
