from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """One follow-up interval on the age/time scale.

    ``entry`` is left-truncation time. An observation enters the risk set only
    when event/censoring time is strictly greater than the current time and
    entry is strictly less than the current time.
    """

    observation_id: str
    group: str
    time: float
    event: bool
    entry: float = 0.0
