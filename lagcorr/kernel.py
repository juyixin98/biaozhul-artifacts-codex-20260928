"""Statistical kernel for irregularly sampled lag correlation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree


RESULT_VERSION = "irregular-lag-pearson-1.0.0"


class InvalidSeriesError(ValueError):
    """The input series cannot form a valid analysis request."""


@dataclass(frozen=True)
class AnalysisConfig:
    detrend: str = "constant"
    standardization: str = "population_zscore"
    matching: str = "symmetric_greedy_nearest_one_to_one"
    min_pairs: int = 8
    min_pair_fraction: float = 0.5
    result_version: str = RESULT_VERSION


@dataclass(frozen=True)
class SeriesInput:
    times: Sequence[float]
    values: Sequence[float]
    name: str = "series"


@dataclass(frozen=True)
class PairRecord:
    a_index: int
    b_index: int
    a_time: float
    b_time: float
    a_value: float
    b_value: float
    aligned_b_time: float
    residual: float


@dataclass(frozen=True)
class LagDiagnosis:
    lag: float
    status: str
    reason_code: str | None
    reason: str | None
    a_count: int
    b_count: int
    candidate_edges: int
    selected_pairs: int
    pair_fraction: float
    selected_a_fraction: float
    selected_b_fraction: float
    selected_time_span: float
    aligned_overlap_span: float
    largest_gap: float
    mean_abs_residual: float | None
    max_abs_residual: float | None
    a_constant: bool
    b_constant: bool
    accepted: bool


@dataclass(frozen=True)
class LagResult:
    lag: float
    status: str
    correlation: float | None
    pair_count: int
    pairs: list[PairRecord]
    diagnosis: LagDiagnosis
    reason_code: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class AnalysisOutcome:
    result_version: str
    config: AnalysisConfig
    tolerance: float
    lags: list[float]
    results: list[LagResult]
    peak_lag: float | None
    peak_correlation: float | None
    peak_kind: str | None
    warning: str

    def to_dict(self) -> dict:
        result = asdict(self)
        result["config"] = asdict(self.config)
        return result


def _prepared_series(series: SeriesInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(series.times) != len(series.values):
        raise InvalidSeriesError(f"{series.name}: times/values length mismatch")
    if len(series.times) == 0:
        raise InvalidSeriesError(f"{series.name}: series must not be empty")

    times = np.asarray(series.times, dtype=np.float64)
    values = np.asarray(series.values, dtype=np.float64)
    if times.ndim != 1 or values.ndim != 1:
        raise InvalidSeriesError(f"{series.name}: times/values must be 1-D")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise InvalidSeriesError(f"{series.name}: times/values must be finite")

    order = np.argsort(times, kind="mergesort")
    sorted_times = times[order]
    if np.any(np.diff(sorted_times) == 0):
        raise InvalidSeriesError(f"{series.name}: timestamps must be unique")
    return sorted_times, values[order], order.astype(np.int64)


def _candidate_edges(
    a_times: np.ndarray,
    shifted_b_times: np.ndarray,
    tolerance: float,
) -> tuple[list[tuple[float, int, int, float]], int]:
    edges: list[tuple[float, int, int, float]] = []
    tree = cKDTree(shifted_b_times[:, None])
    neighbors = tree.query_ball_point(a_times[:, None], r=tolerance)

    for a_idx, nearby in enumerate(neighbors):
        for b_idx in nearby:
            residual = float(shifted_b_times[b_idx] - a_times[a_idx])
            edges.append((abs(residual), a_idx, int(b_idx), residual))

    edges.sort(key=lambda edge: (edge[0], edge[1], edge[2]))
    return edges, sum(len(items) for items in neighbors)


def _select_pairs(
    edges: Iterable[tuple[float, int, int, float]],
) -> list[tuple[int, int, float]]:
    used_a: set[int] = set()
    used_b: set[int] = set()
    selected: list[tuple[int, int, float]] = []

    # Global residual order is deliberately symmetric. Neither dense side is
    # repeatedly consumed by iterating only over points on the other side.
    for _, a_idx, b_idx, residual in edges:
        if a_idx in used_a or b_idx in used_b:
            continue
        used_a.add(a_idx)
        used_b.add(b_idx)
        selected.append((a_idx, b_idx, residual))

    selected.sort(key=lambda item: item[0])
    return selected


def _is_constant(values: np.ndarray) -> bool:
    return bool(np.all(values == values[0]))


def _diagnosis(
    lag: float,
    status: str,
    reason_code: str | None,
    reason: str | None,
    a_times: np.ndarray,
    b_times: np.ndarray,
    shifted_b_times: np.ndarray,
    candidate_edges: int,
    selected: list[tuple[int, int, float]],
    a_constant: bool,
    b_constant: bool,
) -> LagDiagnosis:
    count = len(selected)
    fraction = count / max(1, min(a_times.size, b_times.size))

    if selected:
        a_selected = a_times[[item[0] for item in selected]]
        b_selected = shifted_b_times[[item[1] for item in selected]]
        selected_span = float(
            max(a_selected[-1], b_selected[-1])
            - min(a_selected[0], b_selected[0])
        )
        residuals = np.asarray([item[2] for item in selected])
        mean_abs_residual = float(np.mean(np.abs(residuals)))
        max_abs_residual = float(np.max(np.abs(residuals)))
        selected_timestamps = np.sort(np.concatenate((a_selected, b_selected)))
        largest_gap = float(np.max(np.diff(selected_timestamps))) if selected_timestamps.size > 1 else 0.0
    else:
        selected_span = 0.0
        mean_abs_residual = None
        max_abs_residual = None
        largest_gap = 0.0

    overlap_start = max(float(a_times[0]), float(shifted_b_times[0]))
    overlap_end = min(float(a_times[-1]), float(shifted_b_times[-1]))
    overlap_span = max(0.0, overlap_end - overlap_start)

    return LagDiagnosis(
        lag=float(lag),
        status=status,
        reason_code=reason_code,
        reason=reason,
        a_count=int(a_times.size),
        b_count=int(b_times.size),
        candidate_edges=int(candidate_edges),
        selected_pairs=int(count),
        pair_fraction=float(fraction),
        selected_a_fraction=float(count / max(1, a_times.size)),
        selected_b_fraction=float(count / max(1, b_times.size)),
        selected_time_span=float(selected_span),
        aligned_overlap_span=float(overlap_span),
        largest_gap=float(largest_gap),
        mean_abs_residual=mean_abs_residual,
        max_abs_residual=max_abs_residual,
        a_constant=a_constant,
        b_constant=b_constant,
        accepted=(status == "accepted"),
    )


def analyze_one_lag(
    a_times: np.ndarray,
    a_values: np.ndarray,
    a_indices: np.ndarray,
    b_times: np.ndarray,
    b_values: np.ndarray,
    b_indices: np.ndarray,
    lag: float,
    tolerance: float,
    config: AnalysisConfig,
) -> LagResult:
    shifted_b_times = b_times - float(lag)
    edges, candidate_edges = _candidate_edges(a_times, shifted_b_times, tolerance)
    selected = _select_pairs(edges)

    count = len(selected)
    fraction = count / max(1, min(a_times.size, b_times.size))

    if selected:
        selected_a_positions = [item[0] for item in selected]
        selected_b_positions = [item[1] for item in selected]
        a_constant = _is_constant(a_values[selected_a_positions])
        b_constant = _is_constant(b_values[selected_b_positions])
    else:
        a_constant = False
        b_constant = False

    if count < config.min_pairs:
        status = "unknown"
        reason_code = "INSUFFICIENT_PAIR_COUNT"
        reason = f"fewer than {config.min_pairs} matched pairs: {count}"
    elif fraction < config.min_pair_fraction:
        status = "unknown"
        reason_code = "INSUFFICIENT_PAIR_COVERAGE"
        reason = f"matched fraction {fraction:.3f} is below {config.min_pair_fraction:.3f}"
    elif a_constant or b_constant:
        status = "rejected"
        reason_code = "CONSTANT_SERIES"
        sides = [name for name, flag in (("A", a_constant), ("B", b_constant)) if flag]
        reason = f"constant input series side(s): {', '.join(sides)}"
    else:
        status = "accepted"
        reason_code = None
        reason = None

    diagnosis = _diagnosis(
        lag,
        status,
        reason_code,
        reason,
        a_times,
        b_times,
        shifted_b_times,
        candidate_edges,
        selected,
        a_constant,
        b_constant,
    )

    pairs: list[PairRecord] = [
        PairRecord(
            a_index=int(a_indices[sorted_a]),
            b_index=int(b_indices[sorted_b]),
            a_time=float(a_times[sorted_a]),
            b_time=float(b_times[sorted_b]),
            a_value=float(a_values[sorted_a]),
            b_value=float(b_values[sorted_b]),
            aligned_b_time=float(shifted_b_times[sorted_b]),
            residual=float(residual),
        )
        for sorted_a, sorted_b, residual in selected
    ]
    correlation: float | None = None

    if status == "accepted":
        a_positions = np.asarray(selected_a_positions, dtype=np.int64)
        b_positions = np.asarray(selected_b_positions, dtype=np.int64)
        paired_a = a_values[a_positions]
        paired_b = b_values[b_positions]
        a_z = (paired_a - np.mean(paired_a)) / np.std(paired_a, ddof=0)
        b_z = (paired_b - np.mean(paired_b)) / np.std(paired_b, ddof=0)
        correlation = float(np.clip(np.mean(a_z * b_z), -1.0, 1.0))

    return LagResult(
        lag=float(lag),
        status=status,
        correlation=correlation,
        pair_count=count,
        pairs=pairs,
        diagnosis=diagnosis,
        reason_code=reason_code,
        reason=reason,
    )


def analyze_lags(
    a: SeriesInput,
    b: SeriesInput,
    lags: Sequence[float],
    tolerance: float,
    config: AnalysisConfig | None = None,
) -> AnalysisOutcome:
    config = config or AnalysisConfig()
    if not isinstance(tolerance, (int, float, np.number)):
        raise InvalidSeriesError("tolerance must be a finite positive number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise InvalidSeriesError("tolerance must be a finite positive number")

    clean_lags: list[float] = []
    for lag in lags:
        if not isinstance(lag, (int, float, np.number)) or not math.isfinite(float(lag)):
            raise InvalidSeriesError("all lags must be finite numbers")
        clean_lags.append(float(lag))
    if not clean_lags:
        raise InvalidSeriesError("at least one lag is required")
    if len(set(clean_lags)) != len(clean_lags):
        raise InvalidSeriesError("lags must be unique")

    a = SeriesInput(times=a.times, values=a.values, name="A")
    b = SeriesInput(times=b.times, values=b.values, name="B")
    a_times, a_values, a_indices = _prepared_series(a)
    b_times, b_values, b_indices = _prepared_series(b)

    results = [
        analyze_one_lag(
            a_times,
            a_values,
            a_indices,
            b_times,
            b_values,
            b_indices,
            lag,
            tolerance,
            config,
        )
        for lag in clean_lags
    ]

    accepted = [result for result in results if result.status == "accepted"]
    if accepted:
        peak = min(
            accepted,
            key=lambda result: (-abs(result.correlation or 0.0), result.lag),
        )
        peak_correlation = peak.correlation
        peak_kind = "positive" if (peak_correlation or 0.0) >= 0 else "negative"
    else:
        peak = None
        peak_correlation = None
        peak_kind = None

    return AnalysisOutcome(
        result_version=config.result_version,
        config=config,
        tolerance=tolerance,
        lags=clean_lags,
        results=results,
        peak_lag=None if peak is None else peak.lag,
        peak_correlation=peak_correlation,
        peak_kind=peak_kind,
        warning=(
            "A correlation peak is an alignment statistic only; it does not "
            "establish causation, direction of influence, or lagged effect."
        ),
    )
