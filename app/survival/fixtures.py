"""Deterministic local fixtures for hand calculations and regression tests."""

from __future__ import annotations

from collections.abc import Mapping

from .models import ObservationIn
from ..versions import SYNTHETIC_FIXTURE_VERSION


def obs(time: float, event: bool, group: str, entry_time: float = 0.0) -> ObservationIn:
    return ObservationIn(time=time, event=event, group=group, entry_time=entry_time)


FIXTURES: Mapping[str, list[ObservationIn]] = {
    "hand_km": [
        obs(2.0, True, "device-a"),
        obs(2.0, False, "device-a"),
        obs(3.0, True, "device-a"),
        obs(5.0, False, "device-a"),
        obs(6.0, True, "device-a"),
    ],
    "left_truncation": [
        obs(2.0, True, "delayed", 0.0),
        obs(2.0, False, "delayed", 0.0),
        obs(3.0, True, "delayed", 0.0),
        obs(5.0, False, "delayed", 2.0),
        obs(6.0, True, "delayed", 0.0),
    ],
    "tied_events": [
        obs(4.0, True, "tie-a"),
        obs(4.0, True, "tie-a"),
        obs(4.0, True, "tie-a"),
        obs(4.0, False, "tie-a"),
        obs(5.0, False, "tie-a"),
        obs(6.0, True, "tie-a"),
        obs(7.0, False, "tie-a"),
    ],
    "heavy_censoring": [
        obs(1.0, False, "heavy-censor"),
        obs(2.0, False, "heavy-censor"),
        obs(2.0, False, "heavy-censor"),
        obs(3.0, True, "heavy-censor"),
        obs(4.0, False, "heavy-censor"),
        obs(5.0, False, "heavy-censor"),
    ],
    "all_censored": [
        obs(1.0, False, "all-censor"),
        obs(3.0, False, "all-censor"),
    ],
    "empty_and_all_censored": [
        obs(1.0, False, "all-censor"),
        obs(3.0, False, "all-censor"),
    ],
    "logrank_hand": [
        obs(2.0, True, "A"),
        obs(4.0, True, "A"),
        obs(6.0, False, "A"),
        obs(3.0, True, "B"),
        obs(5.0, True, "B"),
        obs(7.0, False, "B"),
    ],
    "logrank_no_events": [
        obs(2.0, False, "A"),
        obs(4.0, False, "A"),
        obs(3.0, False, "B"),
        obs(5.0, False, "B"),
    ],
}


FIXTURE_GROUPS: Mapping[str, list[str]] = {
    "empty_and_all_censored": ["empty", "all-censor"],
}


def get_fixture(name: str) -> list[ObservationIn]:
    try:
        return [record.model_copy(deep=True) for record in FIXTURES[name]]
    except KeyError as exc:
        available = ", ".join(sorted(FIXTURES))
        raise KeyError(f"unknown fixture {name!r}; available: {available}") from exc


def fixture_version() -> str:
    return SYNTHETIC_FIXTURE_VERSION
