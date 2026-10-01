"""Optional reference-library adapter used by reproducibility checks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .models import ObservationIn


@dataclass(frozen=True)
class ReferenceLibrary:
    name: str
    version: str


def reference_library() -> ReferenceLibrary:
    import lifelines

    return ReferenceLibrary(name="lifelines", version=lifelines.__version__)


def kaplan_meier_reference(
    records: Sequence[ObservationIn],
    group: str,
    *,
    confidence_level: float = 0.95,
):
    """Return lifelines KM rows and metadata for one non-empty group."""

    from lifelines import KaplanMeierFitter

    selected = [record for record in records if record.group == group]
    if not selected:
        raise ValueError("lifelines reference comparison requires a non-empty group")

    durations = np.array([record.time for record in selected], dtype=float)
    events = np.array([record.event for record in selected], dtype=bool)
    entries = np.array([record.entry_time for record in selected], dtype=float)
    fitter = KaplanMeierFitter(alpha=1.0 - confidence_level)
    fitter.fit(durations, events, entry=entries)

    table = fitter.event_table
    survival = fitter.survival_function_.iloc[:, 0]
    ci = fitter.confidence_interval_survival_function_
    rows = []
    for time, event_row in table.iterrows():
        displayed_risk = int(event_row["at_risk"])
        entered_here = int(event_row["entrance"])
        # lifelines displays same-time entrants in at_risk, but its KM
        # recurrence uses at_risk - entrance; the pre-event risk set is
        # therefore adjusted before comparing to our declared ordering.
        pre_event_risk = displayed_risk - entered_here if time > 0 else displayed_risk
        rows.append(
            {
                "time": float(time),
                "entered": int(event_row["entrance"]),
                "at_risk": pre_event_risk,
                "events": int(event_row["observed"]),
                "censored": int(event_row["censored"]),
                "survival": float(survival.loc[time]),
                "ci_lower": float(ci.iloc[:, 0].loc[time]),
                "ci_upper": float(ci.iloc[:, 1].loc[time]),
            }
        )

    median = fitter.median_survival_time_
    return {
        "library": reference_library(),
        "rows": rows,
        "median": None if not np.isfinite(median) else float(median),
    }


def logrank_two_group_reference(records: Sequence[ObservationIn], groups: Sequence[str]):
    """Return the two-group lifelines log-rank statistic and p-value."""

    if len(groups) != 2:
        raise ValueError("lifelines two-group reference expects exactly two groups")

    from lifelines.statistics import logrank_test

    a = [record for record in records if record.group == groups[0]]
    b = [record for record in records if record.group == groups[1]]
    result = logrank_test(
        np.array([r.time for r in a], dtype=float),
        np.array([r.time for r in b], dtype=float),
        np.array([r.event for r in a], dtype=bool),
        np.array([r.event for r in b], dtype=bool),
    )
    return {
        "library": reference_library(),
        "chi_square": float(result.test_statistic),
        "p_value": float(result.p_value),
    }
