"""Log-rank comparison with delayed-entry-compatible risk sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .models import Observation
from .risk import is_at_risk


@dataclass(frozen=True)
class LogRankResult:
    status: str
    reason: str | None
    method: str
    group_order: list[str]
    observed: dict[str, int]
    expected: dict[str, float | None]
    variance_covariance_matrix: list[list[float | None]] | None
    chi_square: float | None
    degrees_freedom: int | None
    p_value: float | None


def _failure(
    groups: list[str],
    reason: str,
    observed: dict[str, int] | None = None,
    expected: dict[str, float | None] | None = None,
) -> LogRankResult:
    return LogRankResult(
        status="failed",
        reason=reason,
        method="log_rank",
        group_order=groups,
        observed=observed or {group: 0 for group in groups},
        expected=expected or {group: None for group in groups},
        variance_covariance_matrix=None,
        chi_square=None,
        degrees_freedom=None,
        p_value=None,
    )


def log_rank(
    observations: list[Observation],
    groups: list[str],
    significance_level: float = 0.05,
) -> LogRankResult:
    """Calculate a multi-sample log-rank statistic.

    At every distinct event time, the common event probability is estimated
    from the fixed risk set containing both events and tied censored
    observations. Left-truncated observations enter only when ``entry < t``.
    The K-group test drops one group and inverts that covariance matrix.
    """

    if not 0.0 < significance_level < 1.0:
        raise ValueError("significance_level must be between 0 and 1")
    if not groups:
        raise ValueError("at least one group must be declared")

    observed = {group: 0 for group in groups}
    expected = {group: 0.0 for group in groups}
    counts_by_group = {group: 0 for group in groups}
    for item in observations:
        if item.group not in observed:
            return _failure(groups, "observation_group_not_declared")
        counts_by_group[item.group] += 1
        if item.event:
            observed[item.group] += 1

    empty_groups = [group for group, count in counts_by_group.items() if count == 0]
    if empty_groups:
        return _failure(
            groups,
            "empty_group_excluded_from_log_rank",
            observed=observed,
            expected={group: None for group in groups},
        )

    if len(groups) == 1:
        return LogRankResult(
            status="not_run",
            reason="log_rank_requires_at_least_two_non_empty_groups",
            method="log_rank",
            group_order=groups,
            observed=observed,
            expected={group: float(observed[group]) for group in groups},
            variance_covariance_matrix=None,
            chi_square=None,
            degrees_freedom=None,
            p_value=None,
        )

    event_times = sorted({item.time for item in observations if item.event})
    total_events = sum(observed.values())
    if total_events == 0:
        return LogRankResult(
            status="not_run",
            reason="no_events_in_any_group",
            method="log_rank",
            group_order=groups,
            observed=observed,
            expected={group: 0.0 for group in groups},
            variance_covariance_matrix=None,
            chi_square=None,
            degrees_freedom=None,
            p_value=None,
        )

    score_groups = groups[:-1]
    dimension = len(score_groups)
    covariance = np.zeros((dimension, dimension), dtype=float)
    score_vector = np.zeros(dimension, dtype=float)

    for time in event_times:
        at_risk = [item for item in observations if is_at_risk(item, time)]
        n_total = len(at_risk)
        events_at_time = sum(1 for item in at_risk if item.time == time and item.event)
        if n_total == 0 or events_at_time == 0:
            continue

        risk_counts = np.array(
            [sum(1 for item in at_risk if item.group == group) for group in score_groups],
            dtype=float,
        )
        event_counts = np.array(
            [
                sum(
                    1
                    for item in at_risk
                    if item.group == group and item.time == time and item.event
                )
                for group in score_groups
            ],
            dtype=float,
        )
        for group in groups:
            n_group = sum(1 for item in at_risk if item.group == group)
            expected[group] += events_at_time * n_group / n_total

        score_vector += event_counts - events_at_time * risk_counts / n_total
        factor = (n_total - events_at_time) / (n_total - 1) if n_total > 1 else 0.0
        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    p_group = risk_counts[i] / n_total
                    covariance[i, i] += factor * events_at_time * p_group * (1.0 - p_group)
                elif j > i:
                    covariance[i, j] -= factor * events_at_time * (
                        risk_counts[i] / n_total
                    ) * (risk_counts[j] / n_total)

    covariance_symmetric = covariance + covariance.T - np.diag(np.diag(covariance))
    matrix_rank = np.linalg.matrix_rank(covariance_symmetric)
    if matrix_rank < dimension:
        matrix = [[float(value) for value in row] for row in covariance_symmetric]
        return LogRankResult(
            status="failed",
            reason="singular_log_rank_variance_matrix",
            method="log_rank",
            group_order=groups,
            observed=observed,
            expected=expected,
            variance_covariance_matrix=matrix,
            chi_square=None,
            degrees_freedom=dimension,
            p_value=None,
        )

    try:
        inverse = np.linalg.inv(covariance_symmetric)
    except np.linalg.LinAlgError:
        matrix = [[float(value) for value in row] for row in covariance_symmetric]
        return LogRankResult(
            status="failed",
            reason="non_invertible_log_rank_variance_matrix",
            method="log_rank",
            group_order=groups,
            observed=observed,
            expected=expected,
            variance_covariance_matrix=matrix,
            chi_square=None,
            degrees_freedom=dimension,
            p_value=None,
        )

    chi_square_statistic = float(score_vector @ inverse @ score_vector)
    p_value = float(chi2.sf(chi_square_statistic, dimension))
    matrix = [[float(value) for value in row] for row in covariance_symmetric]

    return LogRankResult(
        status="estimated",
        reason=None,
        method="log_rank",
        group_order=groups,
        observed=observed,
        expected=expected,
        variance_covariance_matrix=matrix,
        chi_square=chi_square_statistic,
        degrees_freedom=dimension,
        p_value=p_value,
    )
