import pytest

from survival_core import Observation, risk_table

from ._fixtures import delayed_entry_group, simple_group, tie_group


def _row(row):
    return (
        row.time,
        row.kind,
        row.risk_before,
        row.entered,
        row.events,
        row.censored,
        row.risk_after,
    )


def test_basic_risk_table_is_fully_enumerated_by_hand():
    assert [_row(row) for row in risk_table(simple_group())] == [
        (0.0, "baseline", 0, 5, 0, 0, 5),
        (2.0, "event", 5, 0, 1, 0, 4),
        (4.0, "censoring", 4, 0, 0, 1, 3),
        (5.0, "event", 3, 0, 1, 0, 2),
        (6.0, "censoring", 2, 0, 0, 1, 1),
        (8.0, "censoring", 1, 0, 0, 1, 0),
    ]


def test_tied_events_and_censoring_share_one_fixed_risk_set():
    assert [_row(row) for row in risk_table(tie_group())] == [
        (0.0, "baseline", 0, 4, 0, 0, 4),
        (2.0, "event", 4, 0, 2, 1, 1),
        (5.0, "censoring", 1, 0, 0, 1, 0),
    ]


def test_left_truncation_requires_strict_entry_before_time():
    assert [_row(row) for row in risk_table(delayed_entry_group())] == [
        (0.0, "baseline", 0, 1, 0, 0, 1),
            (1.0, "entry", 1, 1, 0, 0, 1),
            (2.0, "event", 2, 1, 1, 0, 1),
            (4.0, "event", 2, 0, 1, 0, 1),
        (5.0, "censoring", 1, 0, 0, 1, 0),
    ]


def test_subject_entering_exactly_at_an_event_time_is_not_at_risk_there():
    observations = [
        Observation("E1", "E", 3.0, True, entry=0.0),
        Observation("E2", "E", 5.0, True, entry=3.0),
    ]
    rows = {row.time: row for row in risk_table(observations)}
    assert (rows[3.0].risk_before, rows[3.0].events, rows[3.0].risk_after) == (1, 1, 0)
    assert (rows[5.0].risk_before, rows[5.0].events, rows[5.0].risk_after) == (1, 1, 0)


@pytest.mark.parametrize(
    ("entry", "time", "query", "expected"),
    [
        (0.0, 5.0, 5.0, True),
        (2.0, 5.0, 2.0, False),
        (2.0, 5.0, 2.000000001, True),
        (2.0, 5.0, 5.0, True),
        (2.0, 5.0, 5.000000001, False),
    ],
)
def test_risk_membership_boundary_predicate(entry, time, query, expected):
    item = Observation("X", "X", time, True, entry=entry)
    from survival_core.risk import is_at_risk

    assert is_at_risk(item, query) is expected
