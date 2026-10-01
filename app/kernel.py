"""Pure statistical kernel for weighted calibration diagnostics."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .contracts import BinRecord, BinningStatistics, CheckResult

SCHEMA_VERSION = "1.0.0"
KERNEL_VERSION = "2026.10.01-weighted-brier-4term"
QUANTILE_TIE_POLICY = (
    "exact probabilities are grouped; the nearest non-splitting group boundary is used; "
    "equidistant boundaries choose the right boundary"
)


def check_result(
    name: str,
    *,
    observed: float,
    expected: float,
    tolerance: float,
    details: dict[str, Any] | None = None,
) -> CheckResult:
    difference = abs(observed - expected)
    return CheckResult(
        name=name,
        passed=bool(math.isfinite(observed) and difference <= tolerance),
        observed=float(observed),
        expected=float(expected),
        absolute_tolerance=float(tolerance),
        details=details or {"absolute_difference": float(difference)},
    )


def fixed_bins(
    p: Sequence[float] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    w: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
    edges: Sequence[float] | np.ndarray | None = None,
    tolerance: float = 1e-10,
) -> BinningStatistics:
    """Compute diagnostics for explicit, fixed boundaries."""

    probabilities, outcomes, weights = _validate_arrays(p, y, w)
    edge_array = (
        np.asarray(edges, dtype=np.float64)
        if edges is not None
        else np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    )
    if len(edge_array) < 2 or edge_array[0] != 0.0 or edge_array[-1] != 1.0:
        raise ValueError("fixed edges must start at 0 and end at 1")
    if np.any(np.diff(edge_array) <= 0.0):
        raise ValueError("fixed edges must be strictly increasing")

    assigned = np.searchsorted(edge_array, probabilities, side="right") - 1
    if np.any(assigned < 0) or np.any(assigned >= len(edge_array) - 1):
        raise ValueError("probabilities lie outside fixed bin edges")

    masks = [assigned == index for index in range(len(edge_array) - 1)]
    ranges = [
        (float(edge_array[index]), float(edge_array[index + 1]))
        for index in range(len(edge_array) - 1)
    ]
    return _statistics(
        probabilities,
        outcomes,
        weights,
        masks=masks,
        ranges=ranges,
        scheme="fixed",
        n_requested=len(edge_array) - 1,
        tie_policy="fixed boundaries; the final edge at p=1 is inclusive",
        tolerance=tolerance,
    )


def equal_frequency_bins(
    p: Sequence[float] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    w: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
    tolerance: float = 1e-10,
) -> BinningStatistics:
    """Partition weighted probability groups without splitting tied values."""

    probabilities, outcomes, weights = _validate_arrays(p, y, w)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    unique_p, inverse = np.unique(probabilities, return_inverse=True)
    group_weights = np.zeros(len(unique_p), dtype=np.float64)
    np.add.at(group_weights, inverse, weights)
    n_groups = len(unique_p)
    cumulative = np.concatenate(([0.0], np.cumsum(group_weights)))
    total_weight = float(cumulative[-1])

    cuts: list[int] = []
    previous = 0
    for quantile in np.linspace(0.0, 1.0, n_bins + 1)[1:-1]:
        target = float(quantile) * total_weight
        candidates = np.arange(previous, n_groups + 1)
        distances = np.abs(cumulative[candidates] - target)
        local_index = int(np.max(np.flatnonzero(distances == np.min(distances))))
        chosen = max(int(candidates[local_index]), previous)
        cuts.append(chosen)
        previous = chosen

    masks: list[np.ndarray] = []
    ranges: list[tuple[float | None, float | None]] = []

    def boundary_value(group_boundary: int) -> float:
        if group_boundary == 0:
            return 0.0
        if group_boundary == n_groups:
            return 1.0
        return float(unique_p[group_boundary])

    for index in range(n_bins):
        lower_group = 0 if index == 0 else cuts[index - 1]
        upper_group = cuts[index] if index < n_bins - 1 else n_groups
        masks.append((inverse >= lower_group) & (inverse < upper_group))
        if lower_group == upper_group:
            value = boundary_value(lower_group)
            ranges.append((value, value))
        else:
            ranges.append((boundary_value(lower_group), boundary_value(upper_group)))

    return _statistics(
        probabilities,
        outcomes,
        weights,
        masks=masks,
        ranges=ranges,
        scheme="equal_frequency",
        n_requested=n_bins,
        tie_policy=QUANTILE_TIE_POLICY,
        tolerance=tolerance,
    )


def _validate_arrays(
    p: Sequence[float] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    w: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(p, dtype=np.float64)
    outcomes = np.asarray(y, dtype=np.int64)
    sample_weights = np.asarray(w, dtype=np.float64)
    if probabilities.ndim != 1 or outcomes.ndim != 1 or sample_weights.ndim != 1:
        raise ValueError("p, y, and w must be one-dimensional")
    if not (len(probabilities) == len(outcomes) == len(sample_weights)):
        raise ValueError("p, y, and w must have equal length")
    if len(probabilities) == 0:
        raise ValueError("at least one sample is required")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("p must be finite and within [0, 1]")
    if np.any((outcomes != 0) & (outcomes != 1)):
        raise ValueError("y must contain only 0 or 1")
    if np.any(~np.isfinite(sample_weights)) or np.any(sample_weights <= 0.0):
        raise ValueError("w must be finite and strictly positive")
    return probabilities, outcomes, sample_weights


def _statistics(
    p: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    masks: list[np.ndarray],
    ranges: list[tuple[float | None, float | None]],
    scheme: str,
    n_requested: int,
    tie_policy: str,
    tolerance: float,
) -> BinningStatistics:
    total_weight = float(np.sum(w))
    total_count = int(len(p))
    effective_sample_size = float(total_weight**2 / np.sum(w**2))
    outcome_rate = float(np.sum(w * y) / total_weight)
    mean_probability = float(np.sum(w * p) / total_weight)
    brier = float(np.sum(w * (p - y) ** 2) / total_weight)

    bin_weight_sum = 0.0
    bin_count_sum = 0
    reliability = 0.0
    resolution = 0.0
    within_bin_variance = 0.0
    records: list[BinRecord] = []
    nonempty = 0

    for index, (mask, (lower, upper)) in enumerate(zip(masks, ranges, strict=True)):
        bin_w = w[mask]
        bin_p = p[mask]
        bin_y = y[mask]
        count = int(mask.sum())
        bin_weight = float(np.sum(bin_w))
        bin_count_sum += count
        bin_weight_sum += bin_weight
        is_empty = bin_weight == 0.0

        if is_empty:
            interval = f"empty[{lower:.17g}, {upper:.17g}]"
            records.append(
                BinRecord(
                    index=index,
                    scheme=scheme,
                    interval=interval,
                    range_lower=lower,
                    range_upper=upper,
                    is_empty=True,
                    count=0,
                    total_weight=0.0,
                    weight_fraction=0.0,
                    effective_sample_size=0.0,
                    p_min=None,
                    p_max=None,
                    p_mean=None,
                    y_mean=None,
                )
            )
            continue

        nonempty += 1
        bin_ess = float(bin_weight**2 / np.sum(bin_w**2))
        bin_p_mean = float(np.sum(bin_w * bin_p) / bin_weight)
        bin_y_mean = float(np.sum(bin_w * bin_y) / bin_weight)
        weight_fraction = bin_weight / total_weight
        reliability += weight_fraction * (bin_p_mean - bin_y_mean) ** 2
        resolution += weight_fraction * (bin_y_mean - outcome_rate) ** 2
        within_bin_variance += weight_fraction * (
            np.sum(bin_w * (bin_p - bin_p_mean) ** 2) / bin_weight
        )
        records.append(
            BinRecord(
                index=index,
                scheme=scheme,
                interval=(
                    f"[{lower:.17g}, {upper:.17g}]"
                    if index == len(masks) - 1
                    else f"[{lower:.17g}, {upper:.17g})"
                ),
                range_lower=lower,
                range_upper=upper,
                is_empty=False,
                count=count,
                total_weight=bin_weight,
                weight_fraction=weight_fraction,
                effective_sample_size=bin_ess,
                p_min=float(np.min(bin_p)),
                p_max=float(np.max(bin_p)),
                p_mean=bin_p_mean,
                y_mean=bin_y_mean,
            )
        )

    uncertainty = float(outcome_rate * (1.0 - outcome_rate))
    relaxed_components = reliability - resolution + uncertainty
    residual = float(brier - (within_bin_variance + relaxed_components))

    weight_check = check_result(
        "total_bin_weight_equals_input_weight",
        observed=bin_weight_sum,
        expected=total_weight,
        tolerance=tolerance,
        details={"input_total_weight": total_weight},
    )
    count_check = check_result(
        "total_bin_count_equals_input_count",
        observed=bin_count_sum,
        expected=total_count,
        tolerance=0.0,
    )
    decomposition_check = check_result(
        "brier_equals_within_variance_plus_reliability_minus_resolution_plus_uncertainty",
        observed=brier,
        expected=within_bin_variance + relaxed_components,
        tolerance=tolerance,
        details={
            "within_bin_forecast_variance": within_bin_variance,
            "reliability": reliability,
            "resolution": resolution,
            "uncertainty": uncertainty,
            "residual": residual,
        },
    )

    return BinningStatistics(
        scheme=scheme,
        n_bins_requested=n_requested,
        n_nonempty_bins=nonempty,
        n_empty_bins=n_requested - nonempty,
        tie_policy=tie_policy,
        sample_count=total_count,
        total_weight=total_weight,
        effective_sample_size=effective_sample_size,
        outcome_rate=outcome_rate,
        mean_probability=mean_probability,
        brier=brier,
        mean_forecast_variance_within_bin=float(within_bin_variance),
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=float(uncertainty),
        relaxed_brier_from_components=float(relaxed_components),
        decomposition_residual=residual,
        bins=records,
        weight_conservation_check=weight_check,
        count_conservation_check=count_check,
        decomposition_check=decomposition_check,
    )
