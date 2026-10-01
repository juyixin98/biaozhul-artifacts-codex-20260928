import pytest

from changepoint_api.errors import UnsupportedConfigurationError
from changepoint_api.experiments.fixtures import (
    mean_step,
    single_outlier,
    tie_fixture,
)
from changepoint_api.experiments.oracle import oracle_optimum
from changepoint_api.kernel import detect_mean_changepoints


def segment_costs(result):
    return [segment.cost for segment in result.segments]


def test_mean_step_has_exact_single_change_and_concrete_costs():
    fixture = mean_step(noise_sd=0.0)
    result = detect_mean_changepoints(
        fixture.samples,
        penalty=0.5,
        min_segment_length=2,
        algorithm="exact_dp",
    )

    assert result.change_points == [10]
    assert result.total_sse == pytest.approx(0.0, abs=1e-18)
    assert result.objective == pytest.approx(1.0, abs=1e-18)
    assert [segment.length for segment in result.segments] == [10, 10]
    assert [segment.mean for segment in result.segments] == [0.0, 5.0]
    assert segment_costs(result) == pytest.approx([0.0, 0.0], abs=1e-18)


def test_moderate_penalty_keeps_a_severe_outlier_as_one_constant_segment():
    fixture = single_outlier(outlier_value=30.0)
    result = detect_mean_changepoints(
        fixture.samples,
        penalty=50.0,
        min_segment_length=1,
        algorithm="exact_dp",
    )

    assert result.change_points == []
    assert len(result.segments) == 1
    assert result.total_sse == pytest.approx(900.0 - 30.0**2 / 20, rel=1e-12)
    assert result.objective == pytest.approx(result.total_sse + 50.0)


def test_zero_penalty_splits_nonconstant_data_to_zero_sse_singletons():
    result = detect_mean_changepoints(
        [0.0, 0.0, 1.0, 1.0],
        penalty=0.0,
        min_segment_length=1,
        algorithm="exact_dp",
    )

    assert result.change_points == [1, 2, 3]
    assert result.total_sse == pytest.approx(0.0, abs=1e-18)
    assert result.objective == pytest.approx(0.0, abs=1e-18)
    assert segment_costs(result) == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_zero_penalty_uses_documented_tie_order_for_constant_data():
    result = detect_mean_changepoints(
        [7.0, 7.0],
        penalty=0.0,
        min_segment_length=1,
        algorithm="exact_dp",
    )

    # Equal objective and SSE are resolved deterministically by more segments.
    assert result.change_points == [1]
    assert result.objective == pytest.approx(0.0, abs=1e-18)


def test_huge_penalty_returns_exact_single_segment():
    fixture = tie_fixture()
    result = detect_mean_changepoints(
        fixture.samples,
        penalty=1.0e30,
        min_segment_length=1,
        algorithm="exact_dp",
    )

    assert result.change_points == []
    assert len(result.segments) == 1
    assert result.segments[0].mean == pytest.approx(0.5)
    assert result.total_sse == pytest.approx(1.0)
    assert result.objective == pytest.approx(1.0 + 1.0e30)


def test_minimum_segment_length_enforces_both_boundaries():
    result = detect_mean_changepoints(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        penalty=0.0,
        min_segment_length=3,
        algorithm="exact_dp",
    )

    assert result.change_points == [3]
    assert [segment.length for segment in result.segments] == [3, 3]
    assert segment_costs(result) == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    ("samples", "penalty", "min_length"),
    [
        ([1.0, -1.0, 0.5, 4.0, 2.0], 0.7, 1),
        ([2.0, 2.0, 8.0, 8.0, 3.0], 1.2, 2),
        ([0.0, 10.0, 0.0, 10.0], 0.0, 1),
        ([0.0, 10.0, 0.0, 10.0], 5.0, 2),
    ],
)
def test_exact_dp_matches_independent_enumeration(
    samples, penalty, min_length
):
    expected = oracle_optimum(samples, penalty, min_length)
    result = detect_mean_changepoints(
        samples,
        penalty=penalty,
        min_segment_length=min_length,
        algorithm="exact_dp",
    )

    assert tuple(result.change_points) == expected.change_points
    assert result.objective == pytest.approx(expected.objective, rel=1e-12)
    assert segment_costs(result) == pytest.approx(
        list(expected.costs), rel=1e-12
    )


def test_minimum_segment_longer_than_sequence_is_rejected_with_category():
    with pytest.raises(UnsupportedConfigurationError) as exc_info:
        detect_mean_changepoints(
            [1.0, 2.0],
            penalty=1.0,
            min_segment_length=3,
            algorithm="exact_dp",
        )

    assert exc_info.value.error_code == "unsupported_configuration"
    assert "no legal segmentation" in str(exc_info.value)
