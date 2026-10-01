"""Shared local synthetic fixtures. These define inputs, not expected outputs."""

from survival_core import Observation


def simple_group() -> list[Observation]:
    return [
        Observation("A1", "A", 2.0, True),
        Observation("A2", "A", 4.0, False),
        Observation("A3", "A", 5.0, True),
        Observation("A4", "A", 6.0, False),
        Observation("A5", "A", 8.0, False),
    ]


def comparison_group_b() -> list[Observation]:
    return [
        Observation("B1", "B", 3.0, True),
        Observation("B2", "B", 7.0, True),
    ]


def tie_group() -> list[Observation]:
    return [
        Observation("T1", "T", 2.0, True),
        Observation("T2", "T", 2.0, True),
        Observation("T3", "T", 2.0, False),
        Observation("T4", "T", 5.0, False),
    ]


def all_censored_group() -> list[Observation]:
    return [
        Observation("C1", "C", 1.0, False),
        Observation("C2", "C", 2.0, False),
        Observation("C3", "C", 3.0, False),
        Observation("C4", "C", 4.0, False),
    ]


def all_event_group() -> list[Observation]:
    return [
        Observation("D1", "D", 1.0, True),
        Observation("D2", "D", 1.0, True),
    ]


def delayed_entry_group() -> list[Observation]:
    return [
        Observation("L1", "L", 2.0, True, entry=1.0),
        Observation("L2", "L", 4.0, True, entry=2.0),
        Observation("L3", "L", 5.0, False, entry=0.0),
    ]
