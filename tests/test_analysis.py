from __future__ import annotations

import pytest

from experiments.reference import reference_detrended_pearson, scenarios
from lag_correlation.kernel.analysis import analyze_lag_correlation
from lag_correlation.kernel.models import Series


@pytest.mark.parametrize("scenario", scenarios(), ids=lambda item: item.name)
def test_synthetic_scenarios_have_concrete_results(scenario) -> None:
    result = analyze_lag_correlation(
        x=Series(scenario.x_name, scenario.x_times, scenario.x_values),
        y=Series(scenario.y_name, scenario.y_times, scenario.y_values),
        min_lag=scenario.lag_grid[0],
        max_lag=scenario.lag_grid[1],
        step=scenario.lag_grid[2],
        tolerance=scenario.tolerance,
        request_id=f"test-{scenario.name}",
    )

    assert result.status == scenario.expected_status
    assert result.failure_category == scenario.expected_failure_category
    assert result.peak_lag == scenario.expected_peak_lag
    assert result.anchor_side == scenario.expected_anchor_side
    assert result.best_lag is not None
    assert result.best_lag.n_pairs == scenario.expected_best_n_pairs
    assert result.best_lag.dense_samples_used == scenario.expected_dense_used
    assert result.best_lag.dense_samples_unmatched == scenario.expected_dense_unmatched

    if scenario.expected_peak_correlation is None:
        assert result.peak_correlation is None
    else:
        assert result.peak_correlation == pytest.approx(
            scenario.expected_peak_correlation, abs=1e-12
        )

    first_pair = (
        result.best_lag.pair_details[0].to_dict()
        if result.best_lag.pair_details
        else {}
    )
    assert first_pair == scenario.expected_first_pair


def test_uniform_special_case_matches_independent_reference() -> None:
    scenario = next(
        item for item in scenarios() if item.name == "uniform_sampling_special_case"
    )
    result = analyze_lag_correlation(
        x=Series(scenario.x_name, scenario.x_times, scenario.x_values),
        y=Series(scenario.y_name, scenario.y_times, scenario.y_values),
        min_lag=scenario.lag_grid[0],
        max_lag=scenario.lag_grid[1],
        step=scenario.lag_grid[2],
        tolerance=scenario.tolerance,
        request_id="test-uniform-reference",
    )
    reference = reference_detrended_pearson(
        scenario.x_times[2:],
        scenario.x_values[2:],
        scenario.y_values[:8],
    )

    assert reference == pytest.approx(1.0)
    assert result.peak_correlation == pytest.approx(reference, abs=1e-12)
    assert result.peak_lag == 2.0


def test_dense_pair_details_use_each_dense_sample_once() -> None:
    scenario = next(
        item for item in scenarios() if item.name == "known_offset_different_density"
    )
    result = analyze_lag_correlation(
        x=Series(scenario.x_name, scenario.x_times, scenario.x_values),
        y=Series(scenario.y_name, scenario.y_times, scenario.y_values),
        min_lag=2.0,
        max_lag=2.0,
        step=1.0,
        tolerance=scenario.tolerance,
        request_id="test-dense-pairs",
    )

    assert result.best_lag is not None
    pairs = result.best_lag.pair_details
    used_indices = [index for pair in pairs for index in pair.paired_indices]
    assert len(pairs) == 10
    assert len(used_indices) == 30
    assert len(set(used_indices)) == 30
    assert [pair.anchor_index for pair in pairs] == list(range(10))
    assert all(len(pair.paired_indices) == 3 for pair in pairs)


def test_sparser_y_is_selected_as_anchor() -> None:
    result = analyze_lag_correlation(
        x=Series("dense", [0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]),
        y=Series("sparse", [0.0, 10.0, 20.0, 30.0, 40.0, 50.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        min_lag=0.0,
        max_lag=0.0,
        step=1.0,
        tolerance=0.1,
        request_id="test-anchor-y",
    )

    assert result.anchor_side == "y"
    assert result.status == "unknown"
    assert result.failure_category == "INSUFFICIENT_OVERLAP"
