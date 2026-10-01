import math

import pytest
from scipy.stats import norm

from survival_core import Observation, kaplan_meier

from ._fixtures import (
    all_censored_group,
    all_event_group,
    delayed_entry_group,
    simple_group,
    tie_group,
)


def _compact(point):
    return (
        point.time,
        point.kind,
        point.risk_before,
        point.events,
        point.censored,
        point.survival,
        point.greenwood_variance,
        point.ci_low,
        point.ci_high,
    )


def test_simple_km_survival_probabilities_and_greenwood_values():
    result = kaplan_meier(simple_group())
    assert result.n == 5
    assert result.events == 2
    assert result.censored == 3
    assert result.median_survival_time is None
    assert result.median_status == "not_reached"
    assert "latest_survival_above_half_median_not_reached" in result.notices

    rows = [(p.time, p.survival, p.greenwood_variance) for p in result.points]
    assert rows[0] == (0.0, 1.0, 0.0)
    assert rows[1] == (2.0, pytest.approx(0.8), pytest.approx(0.032))
    assert rows[2] == (4.0, pytest.approx(0.8), pytest.approx(0.032))
    assert rows[3] == (5.0, pytest.approx(8 / 15), pytest.approx(0.0616296296))
    assert rows[4] == (6.0, pytest.approx(8 / 15), pytest.approx(0.0616296296))
    assert rows[5] == (8.0, pytest.approx(8 / 15), pytest.approx(0.0616296296))


def test_linear_greenwood_interval_is_clipped_at_endpoint_one():
    result = kaplan_meier(simple_group(), confidence_interval_method="greenwood_linear")
    first_event = result.points[1]
    z = norm.ppf(0.975)
    low = 0.8 - z * math.sqrt(0.032)
    assert first_event.ci_low == pytest.approx(low)
    assert first_event.ci_high == 1.0
    assert first_event.ci_clipped is True
    assert "clipped_to_probability_range" in result.notices


def test_log_log_interval_uses_transformed_scale_and_is_not_clipped():
    result = kaplan_meier(simple_group(), confidence_interval_method="log_log")
    point = result.points[1]
    se = math.sqrt(0.032)
    z = norm.ppf(0.975)
    a = math.log(-math.log(0.8))
    scale = se / (0.8 * abs(math.log(0.8)))
    assert point.ci_low == pytest.approx(math.exp(-math.exp(a + z * scale)))
    assert point.ci_high == pytest.approx(math.exp(-math.exp(a - z * scale)))
    assert point.ci_clipped is False


def test_tied_events_are_handled_in_one_product_lift():
    result = kaplan_meier(tie_group())
    assert [(p.time, p.risk_before, p.events, p.censored, p.survival) for p in result.points] == [
        (0.0, 0, 0, 0, 1.0),
        (2.0, 4, 2, 1, 0.5),
        (5.0, 1, 0, 1, 0.5),
    ]
    assert result.points[1].greenwood_variance == pytest.approx(0.0625)
    assert result.median_survival_time == 2.0


def test_all_censored_group_does_not_fabricate_median():
    result = kaplan_meier(all_censored_group())
    assert result.events == 0
    assert result.censored == 4
    assert result.median_survival_time is None
    assert result.median_status == "not_reached"
    assert all(p.survival == 1.0 for p in result.points)
    assert all(p.ci_low == 1.0 and p.ci_high == 1.0 for p in result.points)
    assert "no_events_median_not_reached" in result.notices


def test_empty_group_reports_not_estimable_median():
    result = kaplan_meier([])
    assert result.n == 0
    assert result.median_survival_time is None
    assert result.median_status == "not_estimable"
    assert result.points == []
    assert "empty_group_has_no_median" in result.notices


def test_terminal_all_event_curve_marks_undefined_greenwood_variance():
    result = kaplan_meier(all_event_group())
    point = result.points[1]
    assert point.time == 1.0
    assert point.risk_before == 2
    assert point.events == 2
    assert point.survival == 0.0
    assert point.greenwood_variance is None
    assert point.ci_low == 0.0
    assert point.ci_high == 0.0
    assert result.median_survival_time == 1.0
    assert "boundary_survival_zero_variance_undefined" in result.notices


def test_delayed_entry_km_uses_only_subjects_already_in_risk_set():
    result = kaplan_meier(delayed_entry_group())
    assert [(p.time, p.risk_before, p.entered, p.events, p.survival) for p in result.points] == [
        (0.0, 0, 1, 0, 1.0),
        (1.0, 1, 1, 0, 1.0),
        (2.0, 2, 1, 1, 0.5),
        (4.0, 2, 0, 1, 0.25),
        (5.0, 1, 0, 0, 0.25),
    ]
    assert result.points[2].greenwood_variance == pytest.approx(0.125)
    assert result.points[3].greenwood_variance == pytest.approx(0.0625)


def test_heavy_censoring_after_one_event_keeps_latest_survival_above_half():
    observations = [Observation(f"E", "H", 2.0, True)] + [
        Observation(f"C{i}", "H", float(i), False) for i in range(1, 21)
    ]
    result = kaplan_meier(observations)
    event_row = next(point for point in result.points if point.time == 2.0)
    assert (event_row.risk_before, event_row.events, event_row.censored) == (20, 1, 1)
    assert event_row.survival == pytest.approx(0.95)
    assert result.median_survival_time is None
    assert result.median_status == "not_reached"
    assert "latest_survival_above_half_median_not_reached" in result.notices


def test_invalid_confidence_arguments_are_value_errors():
    with pytest.raises(ValueError, match="confidence_level"):
        kaplan_meier(simple_group(), confidence_level=1.5)
    with pytest.raises(ValueError, match="confidence_interval_method"):
        kaplan_meier(simple_group(), confidence_interval_method="wald")
