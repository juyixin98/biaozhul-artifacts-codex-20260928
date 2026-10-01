from __future__ import annotations

import math

import pytest
from scipy.stats import pearsonr

from lagcorr.kernel import AnalysisConfig, InvalidSeriesError, SeriesInput, analyze_lags


def test_uniform_special_case_matches_independent_pearson():
    a_times = [float(i) for i in range(20)]
    b_times = [i + 3.0 for i in range(20)]
    a_values = [
        0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0,
        4.0, 6.0, 5.0, 5.0, 6.0, 4.0, 7.0, 3.0, 8.0, 2.0,
    ]
    b_values = [
        value + 0.25 * math.sin(index * 1.7)
        for index, value in enumerate(a_values)
    ]

    outcome = analyze_lags(
        SeriesInput(a_times, a_values, "A"),
        SeriesInput(b_times, b_values, "B"),
        lags=[3.0],
        tolerance=0.4,
    )

    result = outcome.results[0]
    expected = float(pearsonr(a_values, b_values).statistic)
    assert result.status == "accepted"
    assert result.pair_count == 20
    assert result.correlation == pytest.approx(expected, abs=1e-12)
    assert [(pair.a_index, pair.b_index) for pair in result.pairs] == [
        (i, i) for i in range(20)
    ]
    assert result.diagnosis.candidate_edges == 20
    assert result.diagnosis.pair_fraction == 1.0


def test_dense_side_cannot_repeat_a_point_and_zero_residual_wins():
    a_times = [2.0 * i for i in range(12)]
    a_values = [float(i) for i in range(12)]
    b_times: list[float] = []
    b_values: list[float] = []
    for i, value in enumerate(a_values):
        b_times.extend((2.0 * i + 1.5, 2.0 * i + 1.0))
        b_values.extend((100.0 + i, value))

    outcome = analyze_lags(
        SeriesInput(a_times, a_values, "A"),
        SeriesInput(b_times, b_values, "B"),
        lags=[1.0],
        tolerance=0.6,
    )

    result = outcome.results[0]
    assert result.status == "accepted"
    assert result.correlation == pytest.approx(1.0)
    assert result.pair_count == 12
    assert result.diagnosis.candidate_edges == 24
    # Every even B index is a 0.5 decoy; odd B index is exact.
    assert [(pair.a_index, pair.b_index) for pair in result.pairs] == [
        (i, 2 * i + 1) for i in range(12)
    ]
    assert len({pair.a_index for pair in result.pairs}) == 12
    assert len({pair.b_index for pair in result.pairs}) == 12


def test_long_gap_is_accepted_and_exposes_gap():
    times = [float(i) for i in range(8)] + [100.0 + i for i in range(8)]
    values = [math.sin(i / 3.0) for i in range(16)]

    outcome = analyze_lags(
        SeriesInput(times, values, "A"),
        SeriesInput([time + 0.25 for time in times], values, "B"),
        lags=[0.25, 80.0],
        tolerance=0.3,
    )

    accepted, displaced = outcome.results
    assert accepted.status == "accepted"
    assert accepted.pair_count == 16
    assert accepted.correlation == pytest.approx(1.0)
    assert accepted.diagnosis.largest_gap > 90.0
    assert displaced.status == "unknown"
    assert displaced.reason_code == "INSUFFICIENT_PAIR_COUNT"


def test_constant_sequence_is_rejected_after_sufficient_matching():
    outcome = analyze_lags(
        SeriesInput([float(i) for i in range(16)], [1.0] * 16, "A"),
        SeriesInput([i + 0.1 for i in range(16)], [2.0] * 16, "B"),
        lags=[0.1],
        tolerance=0.2,
    )

    result = outcome.results[0]
    assert result.status == "rejected"
    assert result.correlation is None
    assert result.reason_code == "CONSTANT_SERIES"
    assert result.diagnosis.selected_pairs == 16
    assert result.diagnosis.a_constant is True
    assert result.diagnosis.b_constant is True


def test_constant_selected_subset_is_rejected_even_if_full_series_varies():
    # At lag 20, only B indices 0..7 are selected. Those selected B values are
    # constant even though later B observations make the full series vary.
    a_times = [float(i) for i in range(8)] + [30.0 + i for i in range(8)]
    a_values = [float(i) for i in range(16)]
    b_times = [20.0 + i for i in range(16)]
    b_values = [7.0] * 8 + [float(i) for i in range(8)]

    outcome = analyze_lags(
        SeriesInput(a_times, a_values, "A"),
        SeriesInput(b_times, b_values, "B"),
        lags=[20.0],
        tolerance=0.1,
    )

    result = outcome.results[0]
    assert result.status == "rejected"
    assert result.reason_code == "CONSTANT_SERIES"
    assert result.correlation is None
    assert result.pair_count == 8
    assert result.diagnosis.a_constant is False
    assert result.diagnosis.b_constant is True


def test_low_coverage_is_unknown_even_though_minimum_count_is_met():
    a_times = [float(i) for i in range(9)] + [30.0 + i for i in range(8)]
    b_times = [float(i) for i in range(8)] + [20.0 + i for i in range(9)]

    outcome = analyze_lags(
        SeriesInput(a_times, [float(i) for i in range(17)], "A"),
        SeriesInput(b_times, [float(i) for i in range(17)], "B"),
        lags=[0.0],
        tolerance=0.6,
    )

    result = outcome.results[0]
    assert result.status == "unknown"
    assert result.reason_code == "INSUFFICIENT_PAIR_COVERAGE"
    assert result.pair_count == 8


def test_coverage_threshold_uses_min_side_and_avoids_dense_weighting():
    a_times = [2.0 * i for i in range(16)]
    b_times = [float(i) for i in range(16)]

    outcome = analyze_lags(
        SeriesInput(a_times, [float(i) for i in range(16)], "A"),
        SeriesInput(b_times, [float(i) for i in range(16)], "B"),
        lags=[0.0],
        tolerance=0.6,
    )

    result = outcome.results[0]
    assert result.status == "accepted"
    assert result.pair_count == 8
    assert result.diagnosis.pair_fraction == 0.5
    assert result.diagnosis.selected_b_fraction == 0.5


@pytest.mark.parametrize(
    ("a_times", "a_values", "b_times", "b_values", "message"),
    [
        ([], [], [1.0], [1.0], "empty"),
        ([1.0, 1.0], [1.0, 2.0], [1.0], [1.0], "unique"),
        ([1.0], [float("nan")], [1.0], [1.0], "finite"),
        ([1.0, 2.0], [1.0], [1.0, 2.0], [1.0, 2.0], "length"),
    ],
)
def test_input_failure_categories(a_times, a_values, b_times, b_values, message):
    with pytest.raises(InvalidSeriesError, match=message):
        analyze_lags(
            SeriesInput(a_times, a_values, "A"),
            SeriesInput(b_times, b_values, "B"),
            lags=[0.0],
            tolerance=0.1,
        )


def test_invalid_lag_and_tolerance_categories():
    series = SeriesInput([1.0, 2.0], [1.0, 2.0], "x")
    with pytest.raises(InvalidSeriesError, match="lag"):
        analyze_lags(series, series, lags=[float("nan")], tolerance=0.1)
    with pytest.raises(InvalidSeriesError, match="tolerance"):
        analyze_lags(series, series, lags=[0.0], tolerance=0.0)


def test_result_version_and_fixed_configuration_are_explicit():
    outcome = analyze_lags(
        SeriesInput([float(i) for i in range(10)], [float(i) for i in range(10)], "A"),
        SeriesInput([float(i) for i in range(10)], [float(i) for i in range(10)], "B"),
        lags=[0.0],
        tolerance=0.1,
    )

    assert outcome.result_version == "irregular-lag-pearson-1.0.0"
    assert outcome.config.detrend == "constant"
    assert outcome.config.standardization == "population_zscore"
    assert outcome.config.matching == "symmetric_greedy_nearest_one_to_one"
    assert "causation" in outcome.warning.lower()
