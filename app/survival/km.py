"""Kaplan-Meier estimator for right-censored, left-truncated records.

Risk-set ordering is intentionally explicit. At each distinct time ``t``:

1. Subjects with ``entry_time < t <= time`` are in the pre-event risk set.
2. Events at ``t`` and right-censored exits at ``t`` are counted against that
   same risk set.
3. Subjects with ``entry_time == t`` enter after those same-time exits for the
   next open interval; the entrance is still shown at this timestamp.
4. Event records update the product-limit estimate; censor-only points leave
   survival unchanged but remain available for audit when requested.

This treats ``entry_time`` as delayed-entry/left-truncation time, not left
censoring. Confidence intervals default to Greenwood variance transformed on
the log(-log S) scale, matching the conventional "log-log" endpoint.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .errors import (
    DuplicateGroupError,
    EmptyDatasetError,
    EmptyGroupError,
    SurvivalValidationError,
    UnknownGroupError,
)
from .models import (
    GroupKMResult,
    KMResult,
    MedianSummary,
    Notice,
    ObservationIn,
    RiskPoint,
    Severity,
)


@dataclass(frozen=True)
class KMCoreOptions:
    confidence_level: float = 0.95
    confidence_interval_method: str = "log-log"
    include_censor_times: bool = True
    include_entry_times: bool = True
    include_baseline: bool = True


def _normalize_groups(
    records: Sequence[ObservationIn], groups: Sequence[str] | None, *, allow_empty: bool
) -> list[str]:
    observed = list(dict.fromkeys(record.group for record in records))
    if groups is None:
        if not records:
            raise EmptyDatasetError("at least one observation is required")
        return observed

    normalized: list[str] = []
    for raw in groups:
        group = raw.strip()
        if not group:
            raise SurvivalValidationError("group labels must not be empty")
        if group in normalized:
            raise DuplicateGroupError(f"duplicate group label: {group}")
        normalized.append(group)

    unknown = sorted(set(observed) - set(normalized))
    if unknown:
        raise UnknownGroupError(f"records contain undeclared group(s): {', '.join(unknown)}")
    if not normalized:
        raise EmptyDatasetError("at least one group is required")
    if not allow_empty and len(records) != len(normalized):
        empty = [group for group in normalized if not any(r.group == group for r in records)]
        if empty:
            raise EmptyGroupError(f"empty declared group(s): {', '.join(empty)}")
    return normalized


def _median_summary(timeline: Sequence[RiskPoint]) -> MedianSummary:
    if not timeline:
        return MedianSummary(
            reachable=False,
            status="empty_group",
            reason="The group contains no observations.",
        )

    event_points = [point for point in timeline if point.event_point]
    if not event_points:
        return MedianSummary(
            reachable=False,
            status="no_events",
            reason="No observed events exist, so the median event time is not estimable.",
        )

    crossing = next((point for point in event_points if point.survival is not None and point.survival <= 0.5), None)
    if crossing is None:
        return MedianSummary(
            reachable=False,
            status="not_reached",
            value=None,
            reason="The Kaplan-Meier estimate does not reach 0.5 during observed follow-up.",
        )

    if crossing.survival == 0.5:
        return MedianSummary(
            reachable=True,
            value=crossing.time,
            interval_lower=crossing.time,
            interval_upper=crossing.time,
            status="estimated",
            reason="The survival estimate equals 0.5 at this observed event time.",
        )

    previous = next((p for p in reversed(event_points) if p.time < crossing.time), None)
    lower = previous.time if previous is not None and previous.survival is not None and previous.survival > 0.5 else 0.0
    return MedianSummary(
        reachable=True,
        value=crossing.time,
        interval_lower=lower,
        interval_upper=crossing.time,
        status="not_identifiable",
        reason=(
            "A discrete KM jump crosses 0.5 without an exact crossing. The "
            "first time with survival at or below 0.5 is reported, with an interval."
        ),
    )


def _confidence_bounds(
    survival: float | None,
    variance: float | None,
    greenwood_terms: float,
    *,
    method: str,
    z: float,
) -> tuple[float | None, float | None, bool]:
    if survival is None or variance is None:
        return None, None, False

    if survival <= 0:
        # Product-limit estimator has reached a terminal zero endpoint.
        # Transform-based intervals are undefined at log(0); do not invent one.
        return 0.0, 0.0, True

    standard_error = float(np.sqrt(max(variance, 0.0)))
    if method == "linear":
        lower = max(0.0, survival - z * standard_error)
        upper = min(1.0, survival + z * standard_error)
        return lower, upper, bool(lower == 0.0 or upper == 1.0)

    if survival >= 1:
        # log(-log(S)) is undefined at S=1. The conventional endpoint is [1, 1].
        return 1.0, 1.0, True

    transformed_se = float(np.sqrt(greenwood_terms) / abs(np.log(survival)))
    log_log = float(np.log(-np.log(survival)))
    lower = float(np.exp(-np.exp(log_log + z * transformed_se)))
    upper = float(np.exp(-np.exp(log_log - z * transformed_se)))
    return min(1.0, max(0.0, lower)), min(1.0, max(0.0, upper)), False


def kaplan_meier_group(
    records: Sequence[ObservationIn],
    group: str,
    options: KMCoreOptions,
) -> tuple[GroupKMResult, list[Notice]]:
    """Fit one named group and return the result plus explanatory notices."""

    group_records = [record for record in records if record.group == group]
    notices: list[Notice] = []
    if not group_records:
        result = GroupKMResult(
            group=group,
            n=0,
            events=0,
            censored=0,
            left_truncated=False,
            timeline=[],
            median=_median_summary([]),
        )
        notices.append(
            Notice(
                code="empty_group",
                severity=Severity.WARNING,
                groups=[group],
                message="Group has no observations; no survival estimate or median was fabricated.",
            )
        )
        return result, notices

    time_values: set[float] = {0.0}
    for record in group_records:
        if record.event or options.include_censor_times:
            time_values.add(float(record.time))
        if record.entry_time > 0 and options.include_entry_times:
            time_values.add(float(record.entry_time))

    entrances: dict[float, int] = defaultdict(int)
    events: dict[float, int] = defaultdict(int)
    censored: dict[float, int] = defaultdict(int)
    for record in group_records:
        entrances[float(record.entry_time)] += 1
        if record.event:
            events[float(record.time)] += 1
        else:
            censored[float(record.time)] += 1

    z = float(norm.ppf(1.0 - (1.0 - options.confidence_level) / 2.0))
    active = 0
    survival = 1.0
    greenwood_terms = 0.0
    timeline: list[RiskPoint] = []
    for time in sorted(time_values):
        entered = entrances.get(time, 0)
        if time == 0:
            active += entered
        d = events.get(time, 0)
        c = censored.get(time, 0)
        at_risk = active

        event_point = d > 0
        point_survival: float | None = survival
        point_variance: float | None = survival**2 * greenwood_terms

        if event_point:
            if at_risk <= 0:
                raise SurvivalValidationError(
                    f"event at time {time} has no subjects in its risk set"
                )
            point_survival = survival * (at_risk - d) / at_risk
            if at_risk == d:
                # Greenwood's terminal d/(n(n-d)) term is undefined; the
                # product-limit endpoint itself is exactly zero.
                point_variance = 0.0
            else:
                greenwood_terms += d / (at_risk * (at_risk - d))
                point_variance = point_survival**2 * greenwood_terms
            survival = point_survival

        lower, upper, endpoint_notice = _confidence_bounds(
            point_survival,
            point_variance,
            greenwood_terms,
            method=options.confidence_interval_method,
            z=z,
        )
        if endpoint_notice:
            notices.append(
                Notice(
                    code="confidence_interval_endpoint",
                    severity=Severity.UNCERTAIN,
                    groups=[group],
                    message=(
                        f"At t={time:g}, the confidence interval uses an explicit "
                        "boundary endpoint because the transformed variance is undefined there."
                    ),
                )
            )

        timeline.append(
            RiskPoint(
                time=time,
                entered=entered,
                at_risk=at_risk,
                events=d,
                censored=c,
                survival=point_survival,
                variance=point_variance,
                ci_lower=lower,
                ci_upper=upper,
                event_point=event_point,
            )
        )
        active -= d + c
        if time > 0:
            active += entered

    if not options.include_baseline:
        timeline = [point for point in timeline if point.time != 0.0]

    median = _median_summary(timeline)
    if median.status in {"not_reached", "no_events", "not_identifiable"}:
        notices.append(
            Notice(
                code=f"median_{median.status}",
                severity=Severity.UNCERTAIN,
                groups=[group],
                message=median.reason,
            )
        )

    result = GroupKMResult(
        group=group,
        n=len(group_records),
        events=sum(events.values()),
        censored=sum(censored.values()),
        left_truncated=any(record.entry_time > 0 for record in group_records),
        timeline=timeline,
        median=median,
    )
    return result, notices


def kaplan_meier(
    records: Iterable[ObservationIn],
    *,
    groups: Sequence[str] | None = None,
    confidence_level: float = 0.95,
    confidence_interval_method: str = "log-log",
    include_censor_times: bool = True,
    include_entry_times: bool = True,
    include_baseline: bool = True,
) -> KMResult:
    """Fit Kaplan-Meier curves for every observed or declared group."""

    record_list = list(records)
    ordered_groups = _normalize_groups(record_list, groups, allow_empty=True)
    options = KMCoreOptions(
        confidence_level=confidence_level,
        confidence_interval_method=confidence_interval_method,
        include_censor_times=include_censor_times,
        include_entry_times=include_entry_times,
        include_baseline=include_baseline,
    )

    results: list[GroupKMResult] = []
    notices: list[Notice] = []
    for group in ordered_groups:
        result, group_notices = kaplan_meier_group(record_list, group, options)
        results.append(result)
        notices.extend(group_notices)

    return KMResult(groups=results, notices=notices)
