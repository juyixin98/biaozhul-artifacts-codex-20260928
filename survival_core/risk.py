"""Risk-set accounting with explicit left truncation and tied endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Observation


@dataclass(frozen=True)
class RiskPoint:
    time: float
    kind: str
    risk_before: int
    entered: int
    events: int
    censored: int
    risk_after: int


def is_at_risk(observation: Observation, time: float) -> bool:
    """Apply the age-scale risk rule.

    A subject is observed just before ``time`` after delayed entry and before
    its terminal time. At exactly ``entry == time`` it has not yet entered, and
    at ``time == observation.time`` it remains at risk while that endpoint is
    processed. Therefore events and censoring at the same timestamp share one
    fixed risk set.
    """

    return observation.entry < time <= observation.time


def _point_kind(events: int, censored: int, entered: int) -> str:
    if events:
        return "event"
    if censored:
        return "censoring"
    return "entry"


def risk_table(observations: list[Observation]) -> list[RiskPoint]:
    """Return one deterministic row per observed timeline transition.

    Rows are sorted numerically by time. Baseline ``time=0`` is always present.
    Positive entry-only times are retained so left truncation can be audited.
    """

    transitions = {0.0} if observations else set()
    for observation in observations:
        transitions.add(observation.time)
        if observation.entry > 0:
            transitions.add(observation.entry)

    points: list[RiskPoint] = []
    for time in sorted(transitions):
        risk_before = sum(1 for item in observations if is_at_risk(item, time))
        entered = sum(
            1
            for item in observations
            if item.entry == time and (time > 0 or item.entry == 0) and item.time > time
        )
        events = sum(1 for item in observations if item.time == time and item.event)
        censored = sum(1 for item in observations if item.time == time and not item.event)
        risk_after = sum(1 for item in observations if item.entry < time < item.time)
        if time == 0:
            risk_before = 0
            entered = sum(1 for item in observations if item.entry == 0)
            risk_after = entered
            kind = "baseline"
        else:
            kind = _point_kind(events, censored, entered)
        points.append(
            RiskPoint(
                time=time,
                kind=kind,
                risk_before=risk_before,
                entered=entered,
                events=events,
                censored=censored,
                risk_after=risk_after,
            )
        )
    return points
