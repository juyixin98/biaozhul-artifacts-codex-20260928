"""Kaplan-Meier survival estimates with explicit endpoint behaviour."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from .models import Observation
from .risk import risk_table


@dataclass(frozen=True)
class KMPoint:
    time: float
    kind: str
    risk_before: int
    entered: int
    events: int
    censored: int
    risk_after: int
    survival: float | None
    greenwood_variance: float | None
    ci_low: float | None
    ci_high: float | None
    ci_clipped: bool


@dataclass(frozen=True)
class KMResult:
    points: list[KMPoint]
    median_survival_time: float | None
    median_status: str
    events: int
    censored: int
    n: int
    notices: list[str]


def _confidence_interval(
    survival: float | None,
    variance: float | None,
    method: str,
    confidence_level: float,
) -> tuple[float | None, float | None, bool, str | None]:
    if survival is None:
        return None, None, False, None

    if survival == 1.0:
        # Before the first event the estimate is a deterministic boundary.
        return 1.0, 1.0, False, "boundary_survival_one"
    if survival == 0.0 or variance is None:
        # Lifelines displays the degenerate endpoint as [0, 0]. We preserve that
        # transparent endpoint while separately marking Greenwood variance null.
        return 0.0, 0.0, False, "boundary_survival_zero_variance_undefined"

    z = norm.ppf((1.0 + confidence_level) / 2.0)
    standard_error = math.sqrt(variance)

    if method == "greenwood_linear":
        low = survival - z * standard_error
        high = survival + z * standard_error
        clipped = bool(low < 0.0 or high > 1.0)
        return max(0.0, min(1.0, low)), max(0.0, min(1.0, high)), clipped, (
            "clipped_to_probability_range" if clipped else None
        )

    if method == "log_log":
        transformed_se = standard_error / (survival * abs(math.log(survival)))
        a = math.log(-math.log(survival))
        low = math.exp(-math.exp(a + z * transformed_se))
        high = math.exp(-math.exp(a - z * transformed_se))
        return low, high, False, None

    raise ValueError(f"unsupported confidence interval method: {method}")


def kaplan_meier(
    observations: list[Observation],
    confidence_level: float = 0.95,
    confidence_interval_method: str = "greenwood_linear",
) -> KMResult:
    """Estimate a Kaplan-Meier curve for one ordered risk table.

    Censoring rows do not multiply the survival probability. The Greenwood
    estimator accumulates ``d / (n * (n - d))`` only at event times. If an
    event removes the final risk set, S becomes exactly zero while Greenwood's
    last summand and normal/log-log intervals are undefined at that endpoint;
    this is reported rather than silently manufactured.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if confidence_interval_method not in {"greenwood_linear", "log_log"}:
        raise ValueError("confidence_interval_method must be greenwood_linear or log_log")

    n = len(observations)
    event_total = sum(1 for item in observations if item.event)
    censor_total = n - event_total
    notices: list[str] = []
    points: list[KMPoint] = []

    survival = 1.0
    greenwood_sum = 0.0
    variance: float | None = 0.0
    terminal_zero = False

    for row in risk_table(observations):
        row_survival: float | None = survival
        row_variance: float | None = variance

        if row.time == 0:
            row_survival = 1.0
            row_variance = 0.0
        elif row.events:
            if row.risk_before <= 0:
                raise ValueError("event encountered outside the risk set")
            survival *= (row.risk_before - row.events) / row.risk_before
            survival = max(0.0, min(1.0, survival))
            denominator = row.risk_before * (row.risk_before - row.events)
            if denominator == 0:
                terminal_zero = True
                row_survival = 0.0
                row_variance = None
            else:
                greenwood_sum += row.events / denominator
                variance = (survival**2) * greenwood_sum
                row_survival = survival
                row_variance = variance
        else:
            row_survival = survival
            row_variance = None if terminal_zero else variance

        low, high, clipped, notice = _confidence_interval(
            row_survival, row_variance, confidence_interval_method, confidence_level
        )
        if notice and not any(existing == notice for existing in notices):
            notices.append(notice)

        points.append(
            KMPoint(
                time=row.time,
                kind=row.kind,
                risk_before=row.risk_before,
                entered=row.entered,
                events=row.events,
                censored=row.censored,
                risk_after=row.risk_after,
                survival=row_survival,
                greenwood_variance=row_variance,
                ci_low=low,
                ci_high=high,
                ci_clipped=clipped,
            )
        )

    median_time: float | None = None
    median_status = "not_reached"
    if n == 0:
        median_status = "not_estimable"
        notices.append("empty_group_has_no_median")
    elif event_total == 0:
        median_status = "not_reached"
        notices.append("no_events_median_not_reached")
    else:
        for point in points:
            if point.kind == "event" and point.survival is not None and point.survival <= 0.5:
                median_time = point.time
                median_status = "estimated"
                break
        if median_time is None:
            median_status = "not_reached"
            notices.append("latest_survival_above_half_median_not_reached")

    return KMResult(
        points=points,
        median_survival_time=median_time,
        median_status=median_status,
        events=event_total,
        censored=censor_total,
        n=n,
        notices=notices,
    )
