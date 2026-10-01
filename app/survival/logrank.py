"""Log-rank comparison with explicit tied-event risk-set handling.

At each distinct event time, subjects with ``entry_time < t <= time`` are in
the pre-event risk set. Subjects entering at ``t`` become available only after
same-time exits. Censored subjects at the same time are in the pre-event risk
set. Expected counts use the common hypergeometric formula ``n_g d / n``
(Breslow/full-cohort handling of ties), and the statistic is the omnibus
chi-square statistic using all but the last sorted group.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy.stats import chi2

from .errors import (
    DuplicateGroupError,
    EmptyDatasetError,
    EmptyGroupError,
    InsufficientGroupsError,
    SurvivalValidationError,
    UnknownGroupError,
)
from .models import (
    EventContribution,
    LogRankGroupStatistic,
    LogRankRequest,
    LogRankResult,
    Notice,
    ObservationIn,
    Severity,
)


def _declared_groups(
    records: Sequence[ObservationIn], declared: Sequence[str] | None
) -> list[str]:
    observed = list(dict.fromkeys(record.group for record in records))
    if not records:
        raise EmptyDatasetError("at least one observation is required")

    if declared is None:
        if len(observed) < 2:
            raise InsufficientGroupsError("log-rank comparison requires at least two groups")
        return observed

    groups: list[str] = []
    for raw in declared:
        group = raw.strip()
        if not group:
            raise SurvivalValidationError("group labels must not be empty")
        if group in groups:
            raise DuplicateGroupError(f"duplicate group label: {group}")
        groups.append(group)

    unknown = sorted(set(observed) - set(groups))
    if unknown:
        raise UnknownGroupError(f"records contain undeclared group(s): {', '.join(unknown)}")
    if len(groups) < 2:
        raise InsufficientGroupsError("log-rank comparison requires at least two groups")
    empty = [group for group in groups if not any(record.group == group for record in records)]
    if empty:
        raise EmptyGroupError(f"empty declared group(s): {', '.join(empty)}")
    return groups


def _risk_counts(
    records: Sequence[ObservationIn], groups: Sequence[str], time: float
) -> np.ndarray:
    return np.array(
        [
            sum(
                record.entry_time < time <= record.time
                for record in records
                if record.group == group
            )
            for group in groups
        ],
        dtype=float,
    )


def logrank_test(
    records: Iterable[ObservationIn], *, groups: Sequence[str] | None = None
) -> LogRankResult:
    """Calculate an omnibus log-rank test for two or more groups."""

    record_list = list(records)
    ordered_groups = _declared_groups(record_list, groups)
    event_times = sorted({float(r.time) for r in record_list if r.event})

    observed_total = np.zeros(len(ordered_groups), dtype=float)
    expected_total = np.zeros(len(ordered_groups), dtype=float)
    contributions: list[EventContribution] = []
    covariance = np.zeros((len(ordered_groups), len(ordered_groups)), dtype=float)

    for time in event_times:
        n_group = _risk_counts(record_list, ordered_groups, time)
        total_risk = float(n_group.sum())
        d_group = np.array(
            [
                sum(r.event and r.time == time and r.group == group for r in record_list)
                for group in ordered_groups
            ],
            dtype=float,
        )
        total_events = float(d_group.sum())
        if total_risk <= 0:
            raise SurvivalValidationError(f"event at t={time:g} has an empty risk set")

        expected = n_group * total_events / total_risk
        observed_total += d_group
        expected_total += expected

        if total_risk > 1:
            factor = (
                total_events
                * (total_risk - total_events)
                / (total_risk**2 * (total_risk - 1.0))
            )
            covariance += factor * (np.diag(n_group * total_risk) - np.outer(n_group, n_group))

        for index, group in enumerate(ordered_groups):
            contributions.append(
                EventContribution(
                    time=time,
                    group=group,
                    observed=int(d_group[index]),
                    expected=float(expected[index]),
                    variance_diagonal=float(covariance[index, index]),
                )
            )

    z = observed_total - expected_total
    reduced_covariance = covariance[:-1, :-1]
    reduced_z = z[:-1]
    total_events = int(observed_total.sum())

    notices: list[Notice] = []
    testable = total_events > 0 and bool(np.linalg.matrix_rank(reduced_covariance) >= len(reduced_z))
    if total_events == 0:
        notices.append(
            Notice(
                code="no_events",
                severity=Severity.UNCERTAIN,
                groups=ordered_groups,
                message="All observations are right-censored; no log-rank comparison is defined.",
            )
        )
    elif not testable:
        notices.append(
            Notice(
                code="singular_logrank_variance",
                severity=Severity.UNCERTAIN,
                groups=ordered_groups,
                message=(
                    "Observed events do not provide full rank information across "
                    "groups; a conventional log-rank p-value is not reported."
                ),
            )
        )

    if testable:
        chi_square = float(reduced_z @ np.linalg.solve(reduced_covariance, reduced_z))
        p_value = float(chi2.sf(chi_square, len(ordered_groups) - 1))
        status = "valid"
    else:
        chi_square = 0.0
        p_value = None
        status = "not_testable"

    return LogRankResult(
        groups=ordered_groups,
        degrees_of_freedom=len(ordered_groups) - 1,
        chi_square=chi_square,
        p_value=p_value,
        testable=testable,
        status=status,
        event_contributions=contributions,
        group_statistics=[
            LogRankGroupStatistic(
                group=group,
                observed=int(observed_total[index]),
                expected=float(expected_total[index]),
                observed_minus_expected=float(z[index]),
            )
            for index, group in enumerate(ordered_groups)
        ],
        notices=notices,
    )


def logrank_from_request(request: LogRankRequest) -> LogRankResult:
    return logrank_test(request.observations)
