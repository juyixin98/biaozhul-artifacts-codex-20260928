from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.survival.fixtures import get_fixture
from app.survival.km import kaplan_meier
from app.survival.models import ObservationIn


def obs(time, event, group="g", entry=0.0):
    return ObservationIn(time=time, event=event, group=group, entry_time=entry)


def _point(result, time):
    timeline = getattr(result, "timeline", None)
    if timeline is None:
        timeline = result.groups[0].timeline
    return next(point for point in timeline if point.time == time)


def test_hand_km_tied_event_censor_table_is_explicit():
    result = kaplan_meier(get_fixture("hand_km"))
    group = result.groups[0]

    assert group.n == 5
    assert group.events == 3
    assert group.censored == 2
    assert [
        (p.time, p.entered, p.at_risk, p.events, p.censored, p.survival)
        for p in group.timeline
    ] == [
        (0.0, 5, 5, 0, 0, 1.0),
        (2.0, 0, 5, 1, 1, 0.8),
        (3.0, 0, 3, 1, 0, pytest.approx(8 / 15)),
        (5.0, 0, 2, 0, 1, pytest.approx(8 / 15)),
        (6.0, 0, 1, 1, 0, 0.0),
    ]

    assert group.median.reachable is True
    assert group.median.value == 6.0
    assert group.median.interval_lower == 3.0
    assert group.median.interval_upper == 6.0
    assert group.median.status == "not_identifiable"


def test_left_truncation_risk_set_matches_strict_interval_order():
    result = kaplan_meier(get_fixture("left_truncation"))
    group = result.groups[0]

    assert [
        (p.time, p.entered, p.at_risk, p.events, p.censored, p.survival)
        for p in group.timeline
    ] == [
        (0.0, 4, 4, 0, 0, 1.0),
        (2.0, 1, 4, 1, 1, 0.75),
        (3.0, 0, 3, 1, 0, 0.5),
        (5.0, 0, 2, 0, 1, 0.5),
        (6.0, 0, 1, 1, 0, 0.0),
    ]


def test_tied_events_use_common_risk_set():
    result = kaplan_meier(get_fixture("tied_events"))
    group = result.groups[0]

    assert _point(group, 4.0).at_risk == 7
    assert _point(group, 4.0).events == 3
    assert _point(group, 4.0).censored == 1
    assert _point(group, 4.0).survival == pytest.approx(4 / 7)
    assert _point(group, 5.0).survival == pytest.approx(4 / 7)
    assert _point(group, 6.0).at_risk == 2
    assert _point(group, 6.0).survival == pytest.approx(2 / 7)


def test_heavy_and_all_censoring_do_not_fabricate_median():
    heavy = kaplan_meier(get_fixture("heavy_censoring")).groups[0]
    assert [(p.time, p.at_risk, p.events, p.censored, p.survival) for p in heavy.timeline] == [
        (0.0, 6, 0, 0, 1.0),
        (1.0, 6, 0, 1, 1.0),
        (2.0, 5, 0, 2, 1.0),
        (3.0, 3, 1, 0, 2 / 3),
        (4.0, 2, 0, 1, 2 / 3),
        (5.0, 1, 0, 1, 2 / 3),
    ]
    assert heavy.median.reachable is False
    assert heavy.median.value is None
    assert heavy.median.status == "not_reached"

    all_censored = kaplan_meier(get_fixture("all_censored")).groups[0]
    assert all(point.survival == 1.0 for point in all_censored.timeline)
    assert all_censored.median.reachable is False
    assert all_censored.median.value is None
    assert all_censored.median.status == "no_events"


def test_empty_group_returns_empty_curve_not_a_fake_median():
    records = get_fixture("empty_and_all_censored")
    result = kaplan_meier(records, groups=["empty", "all-censor"])

    empty = next(group for group in result.groups if group.group == "empty")
    assert empty.n == 0
    assert empty.timeline == []
    assert empty.median.value is None
    assert empty.median.status == "empty_group"
    assert {notice.code for notice in result.notices if "empty" in notice.groups} == {"empty_group"}


def test_exact_half_survival_has_median_at_event_time():
    records = [
        obs(1.0, True),
        obs(2.0, True),
        obs(3.0, False),
        obs(4.0, False),
    ]
    group = kaplan_meier(records).groups[0]
    assert _point(group, 2.0).survival == 0.5
    assert group.median.status == "estimated"
    assert group.median.value == 2.0


def test_loglog_and_linear_confidence_methods_are_distinct_and_bounded():
    records = get_fixture("hand_km")
    loglog = kaplan_meier(records, confidence_interval_method="log-log").groups[0]
    linear = kaplan_meier(records, confidence_interval_method="linear").groups[0]

    assert _point(loglog, 2.0).ci_lower == pytest.approx(0.203809263268, abs=1e-10)
    assert _point(loglog, 2.0).ci_upper == pytest.approx(0.969179788867, abs=1e-10)
    assert _point(linear, 2.0).ci_lower == pytest.approx(0.4493909810)
    assert _point(linear, 2.0).ci_upper == pytest.approx(1.0)
    assert _point(loglog, 0.0).ci_lower == _point(loglog, 0.0).ci_upper == 1.0
    assert _point(loglog, 6.0).ci_lower == _point(loglog, 6.0).ci_upper == 0.0


def test_same_time_late_entrant_enters_after_same_time_exits():
    records = [
        obs(2.0, True, group="x"),
        obs(3.0, True, group="x", entry=2.0),
    ]
    result = kaplan_meier(records, include_entry_times=True).groups[0]
    assert [(p.time, p.entered, p.at_risk, p.events, p.survival) for p in result.timeline] == [
        (0.0, 1, 1, 0, 1.0),
        (2.0, 1, 1, 1, 0.0),
        (3.0, 0, 1, 1, 0.0),
    ]


def test_invalid_zero_duration_left_truncation_has_specific_failure_category():
    with pytest.raises(ValidationError) as exc_info:
        ObservationIn(time=2.0, event=True, group="g", entry_time=2.0)
    message = str(exc_info.value)
    assert "entry_time must be strictly smaller than time" in message


def test_negative_times_and_invalid_confidence_are_contract_failures():
    with pytest.raises(ValidationError):
        ObservationIn(time=-1.0, event=True, group="g")
    with pytest.raises(ValidationError):
        ObservationIn(time=0.0, event=True, group="g")


def test_greenwood_variance_matches_hand_value():
    result = kaplan_meier(get_fixture("hand_km")).groups[0]
    assert _point(result, 2.0).variance == pytest.approx(0.032)
    assert _point(result, 3.0).variance == pytest.approx(0.0616296296296)
    assert math.isfinite(_point(result, 6.0).variance)
