"""Internal immutable kernel result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Series:
    name: str
    times: list[float]
    values: list[float]


@dataclass(frozen=True)
class PairDetail:
    anchor_side: Literal["x", "y"]
    anchor_time: float
    paired_time: float
    anchor_index: int
    paired_indices: tuple[int, ...]
    time_delta: float
    lag: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_side": self.anchor_side,
            "anchor_time": self.anchor_time,
            "paired_time": self.paired_time,
            "anchor_index": self.anchor_index,
            "paired_indices": list(self.paired_indices),
            "time_delta": self.time_delta,
            "lag": self.lag,
        }


@dataclass(frozen=True)
class LagResult:
    lag: float
    n_pairs: int
    correlation: float | None
    accepted: bool
    status: str
    reason: str
    pair_count_from_anchor: int
    dense_samples_used: int
    dense_samples_unmatched: int
    pair_details: tuple[PairDetail, ...] = field(default_factory=tuple)

    def to_dict(self, include_pairs: bool = False) -> dict[str, Any]:
        result = {
            "lag": self.lag,
            "n_pairs": self.n_pairs,
            "correlation": self.correlation,
            "accepted": self.accepted,
            "status": self.status,
            "reason": self.reason,
            "pair_count_from_anchor": self.pair_count_from_anchor,
            "dense_samples_used": self.dense_samples_used,
            "dense_samples_unmatched": self.dense_samples_unmatched,
        }
        if include_pairs:
            result["pair_details"] = [pair.to_dict() for pair in self.pair_details]
        else:
            result["pair_details"] = None
        return result


@dataclass(frozen=True)
class AnalysisResult:
    request_id: str
    status: Literal["ok", "unknown"]
    failure_category: str | None
    reason: str
    peak_lag: float | None
    peak_correlation: float | None
    anchor_side: Literal["x", "y"]
    anchor_median_interval: float
    other_median_interval: float
    minimum_pairs: int
    detrend: str
    standardization: str
    valid_lag_count: int
    evaluated_lag_count: int
    best_lag: LagResult | None
    diagnostics: tuple[dict[str, Any], ...]
