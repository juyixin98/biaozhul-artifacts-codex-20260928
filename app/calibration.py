"""Weighted calibration models fit only on the declared training partition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.interpolate import interp1d

from .errors import ContractViolation


class Calibrator(Protocol):
    type: str

    def transform(self, p: np.ndarray) -> np.ndarray: ...

    def parameters(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class IdentityCalibrator:
    type: str = "identity"

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=np.float64)

    def parameters(self) -> dict[str, Any]:
        return {"description": "probabilities are passed through unchanged"}


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Linear interpolation between adjacent non-decreasing PAVA blocks."""

    predictor: interp1d
    blocks: tuple[dict[str, float], ...]
    nodes: tuple[float, ...]
    values: tuple[float, ...]
    type: str = "isotonic"

    def transform(self, p: np.ndarray) -> np.ndarray:
        values = np.asarray(p, dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise ContractViolation(
                "calibrator input probability must be finite and within [0, 1]",
                code="probability_out_of_bounds",
            )
        result = np.asarray(self.predictor(values), dtype=np.float64)
        if np.any(result < 0.0) or np.any(result > 1.0):
            raise ContractViolation(
                "calibrator produced a probability outside [0, 1]",
                code="calibrated_probability_out_of_bounds",
            )
        return result

    def parameters(self) -> dict[str, Any]:
        return {
            "method": "weighted_pool_adjacent_violators_with_linear_interpolation",
            "nodes": list(self.nodes),
            "values": list(self.values),
            "blocks": list(self.blocks),
        }


def _aggregate_equal_probabilities(
    p: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    changes = np.flatnonzero(np.diff(sorted_p)) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(sorted_p)]

    unique_p = sorted_p[starts]
    weights = np.array(
        [np.sum(w[order[start:end]]) for start, end in zip(starts, ends)],
        dtype=np.float64,
    )
    events = np.array(
        [np.sum(y[order[start:end]] * w[order[start:end]]) for start, end in zip(starts, ends)],
        dtype=np.float64,
    )
    return unique_p, weights, events / weights


def fit_isotonic(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> IsotonicCalibrator:
    """Fit weighted isotonic regression with the PAVA algorithm."""

    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    unique_p, group_w, group_y = _aggregate_equal_probabilities(p, y, w)

    blocks: list[dict[str, float]] = []
    for probability, weight, event_rate in zip(unique_p, group_w, group_y, strict=True):
        block = {
            "lower_p": float(probability),
            "upper_p": float(probability),
            "weight": float(weight),
            "event_rate": float(event_rate),
        }
        while blocks and blocks[-1]["event_rate"] > block["event_rate"]:
            previous = blocks.pop()
            merged_weight = previous["weight"] + block["weight"]
            merged_rate = (
                previous["weight"] * previous["event_rate"]
                + block["weight"] * block["event_rate"]
            ) / merged_weight
            block = {
                "lower_p": previous["lower_p"],
                "upper_p": block["upper_p"],
                "weight": merged_weight,
                "event_rate": merged_rate,
            }
        blocks.append(block)

    nodes = np.asarray([block["upper_p"] for block in blocks], dtype=np.float64)
    values = np.asarray([block["event_rate"] for block in blocks], dtype=np.float64)
    predictor = interp1d(
        nodes,
        values,
        kind="linear",
        bounds_error=False,
        fill_value=(float(values[0]), float(values[-1])),
        assume_sorted=True,
    )
    serializable_blocks = tuple(
        {key: float(value) for key, value in block.items()} for block in blocks
    )
    return IsotonicCalibrator(
        predictor=predictor,
        blocks=serializable_blocks,
        nodes=tuple(float(x) for x in nodes),
        values=tuple(float(x) for x in values),
    )


def make_calibrator(
    calibrator_type: str, p: np.ndarray, y: np.ndarray, w: np.ndarray
) -> Calibrator:
    if calibrator_type == "identity":
        return IdentityCalibrator()
    if calibrator_type == "isotonic":
        return fit_isotonic(p, y, w)
    raise ContractViolation(
        f"unsupported calibrator type: {calibrator_type}", code="unsupported_calibrator"
    )
