from __future__ import annotations

import pytest

from app.survival.errors import InsufficientGroupsError
from app.survival.fixtures import get_fixture
from app.survival.logrank import logrank_test


def test_hand_two_group_logrank_values_and_contributions():
    result = logrank_test(get_fixture("logrank_hand"))

    assert result.groups == ["A", "B"]
    assert result.degrees_of_freedom == 1
    assert result.chi_square == pytest.approx(0.07390300230946893, abs=1e-14)
    assert result.p_value == pytest.approx(0.7857365379599127, abs=1e-14)
    assert [(stat.group, stat.observed, round(stat.expected, 4)) for stat in result.group_statistics] == [
        ("A", 2, 1.7333),
        ("B", 2, 2.2667),
    ]

    at_two = [row for row in result.event_contributions if row.time == 2.0]
    assert [(row.group, row.observed, row.expected) for row in at_two] == [
        ("A", 1, 0.5),
        ("B", 0, 0.5),
    ]


def test_all_censored_logrank_is_uncertain_without_fake_p_value():
    result = logrank_test(get_fixture("logrank_no_events"))
    assert result.testable is False
    assert result.status == "not_testable"
    assert result.chi_square == 0.0
    assert result.p_value is None
    assert result.notices[0].code == "no_events"


def test_single_group_has_specific_failure_category():
    records = get_fixture("hand_km")
    with pytest.raises(InsufficientGroupsError) as exc_info:
        logrank_test(records)
    assert exc_info.value.code == "insufficient_groups"


def test_tied_event_logrank_uses_complete_risk_set():
    records = [
        *get_fixture("hand_km"),
    ]
    # At t=2, both groups have one subject at risk; both events occur at t=2.
    for record in records:
        record.group = "a"
    records[0].group = "a"
    other = get_fixture("hand_km")
    for record in other:
        record.group = "b"
        record.time = 3.0 if record.time == 2.0 else record.time + 1
    records.extend(other)

    result = logrank_test(records)
    t2 = {row.group: row for row in result.event_contributions if row.time == 2.0}
    assert t2["a"].expected == pytest.approx(0.5)
    assert t2["b"].expected == pytest.approx(0.5)


def test_left_truncation_risk_count_excludes_same_time_entrant():
    records = [
        make_obs(time=2.0, event=True, group="a"),
        make_obs(time=4.0, event=False, group="a"),
        make_obs(time=2.0, event=False, group="b"),
        make_obs(time=3.0, event=True, group="b", entry_time=2.0),
        make_obs(time=5.0, event=False, group="b"),
    ]
    result = logrank_test(records)
    t2 = {row.group: row for row in result.event_contributions if row.time == 2.0}
    assert t2["a"].expected == 1.0
    assert t2["b"].expected == 0.0


def make_obs(time, event, group, entry_time=0.0):
    from app.survival.models import ObservationIn

    return ObservationIn(time=time, event=event, group=group, entry_time=entry_time)
