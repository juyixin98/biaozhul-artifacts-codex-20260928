"""Agreement tests against an independently implemented package.

These expected values are produced here from lifelines, never from
``survival_core``. Hand constants in the other test files remain the primary
specification; these tests guard implementation drift against a second source.
"""

import pytest

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from survival_core import kaplan_meier, log_rank

from ._fixtures import (
    all_event_group,
    comparison_group_b,
    delayed_entry_group,
    simple_group,
    tie_group,
)


def _reference_curve(observations):
    fitter = KaplanMeierFitter()
    fitter.fit(
        [item.time for item in observations],
        event_observed=[int(item.event) for item in observations],
        entry=[item.entry for item in observations],
    )
    return fitter


@pytest.mark.parametrize(
    "fixture_factory",
    [simple_group, tie_group, all_event_group, delayed_entry_group],
)
def test_km_survival_matches_lifelines_at_event_times(fixture_factory):
    observations = fixture_factory()
    result = kaplan_meier(observations, confidence_interval_method="log_log")
    reference = _reference_curve(observations)

    reference_survival = dict(
        zip(
            reference.survival_function_.index,
            reference.survival_function_.iloc[:, 0],
            strict=True,
        )
    )
    for point in result.points:
        if point.kind != "event":
            continue
        assert point.survival == pytest.approx(reference_survival[point.time])


def test_km_loglog_intervals_match_lifelines():
    observations = simple_group()
    result = kaplan_meier(observations, confidence_interval_method="log_log")
    reference = _reference_curve(observations)
    for point in result.points:
        if point.kind != "event":
            continue
        assert point.ci_low == pytest.approx(
            reference.confidence_interval_.loc[point.time].iloc[0]
        )
        assert point.ci_high == pytest.approx(
            reference.confidence_interval_.loc[point.time].iloc[1]
        )


def test_two_group_logrank_matches_lifelines_pairwise_api():
    observations = simple_group() + comparison_group_b()
    result = log_rank(observations, ["A", "B"])
    reference = logrank_test(
        [item.time for item in simple_group()],
        [item.time for item in comparison_group_b()],
        [int(item.event) for item in simple_group()],
        [int(item.event) for item in comparison_group_b()],
    )
    assert result.chi_square == pytest.approx(reference.test_statistic)
    assert result.p_value == pytest.approx(reference.p_value)


def test_delayed_entry_logrank_matches_lifelines_entry_arguments():
    observations = [
        *_transform_delayed_fixture("A"),
        *_transform_delayed_fixture("B"),
    ]
    result = log_rank(observations, ["A", "B"])
    a = [item for item in observations if item.group == "A"]
    b = [item for item in observations if item.group == "B"]
    reference = logrank_test(
        [item.time for item in a],
        [item.time for item in b],
        [int(item.event) for item in a],
        [int(item.event) for item in b],
        entry_A=[item.entry for item in a],
        entry_B=[item.entry for item in b],
    )
    assert result.chi_square == pytest.approx(reference.test_statistic)
    assert result.p_value == pytest.approx(reference.p_value)


def test_multivariate_logrank_matches_lifelines_three_groups():
    observations = [
        *_transform_delayed_fixture("A"),
        *_transform_delayed_fixture("B"),
        *_transform_delayed_fixture("C"),
    ]
    result = log_rank(observations, ["A", "B", "C"])
    reference = multivariate_logrank_test(
        [item.time for item in observations],
        [item.group for item in observations],
        [int(item.event) for item in observations],
        entry=[item.entry for item in observations],
    )
    assert result.chi_square == pytest.approx(reference.test_statistic)
    assert result.p_value == pytest.approx(reference.p_value)
    assert result.degrees_freedom == 2


def _transform_delayed_fixture(group: str):
    from survival_core import Observation

    base = delayed_entry_group()
    return [
        Observation(
            observation_id=f"{group}-{item.observation_id}",
            group=group,
            time=item.time + (0.0 if group == "A" else 1.2),
            event=item.event,
            entry=item.entry,
        )
        for item in base
    ]
