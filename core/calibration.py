"""Fitted probability calibration and separate evaluation handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binning import assign_bins, resolve_edges
from .errors import EmptyCalibrationBin


@dataclass(frozen=True)
class HistogramCalibrator:
    """Deterministic histogram recalibrator fitted only on training data."""

    edges: np.ndarray
    bin_predicted_rates: tuple[float | None, ...]

    @classmethod
    def fit(cls, train_p: np.ndarray, train_y: np.ndarray, train_w: np.ndarray, method: str, n_bins: int, explicit_edges: list[float] | None = None) -> "HistogramCalibrator":
        edges = resolve_edges(train_p, train_w, method, n_bins, explicit_edges)
        assignments = assign_bins(train_p, edges)
        rates: list[float | None] = []
        for index in range(len(edges) - 1):
            selected = assignments == index
            weight = float(np.sum(train_w[selected]))
            if weight <= 0.0:
                rates.append(None)
            else:
                rates.append(float(np.sum(train_y[selected] * train_w[selected]) / weight))
        return cls(edges=edges, bin_predicted_rates=tuple(rates))

    def transform(self, p: np.ndarray) -> np.ndarray:
        assignments = assign_bins(p, self.edges)
        output = np.zeros_like(p, dtype=float)
        for index in np.unique(assignments):
            rate = self.bin_predicted_rates[int(index)]
            if rate is None:
                lower = float(self.edges[index])
                upper = float(self.edges[index + 1])
                raise EmptyCalibrationBin(
                    f"evaluation probability falls into an empty fitted bin "
                    f"{index} with interval [{lower}, {upper})"
                )
            output[assignments == index] = rate
        return output
