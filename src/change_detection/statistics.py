from __future__ import annotations

import math

import numpy as np

from .errors import InvalidParameterError, InvalidSeriesError, NumericStabilityError


class StablePrefixStats:
    """Length, mean, and Welford M2 prefix statistics.

    Segment variance is combined with Chan's parallel-variance identity.
    The M2 accumulator avoids the catastrophic ``sum(x^2) - n * mean^2``
    formulation. Inputs are checked before any optimization work starts.
    """

    def __init__(self, values: np.ndarray | list[float]):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise InvalidSeriesError("values must be a one-dimensional sequence")
        if array.size == 0:
            raise InvalidSeriesError("values must contain at least one observation")
        if not np.all(np.isfinite(array)):
            raise InvalidSeriesError("all values must be finite")

        self.values = array
        self.n = int(array.size)
        self.count = np.zeros(self.n + 1, dtype=np.int64)
        self.mean = np.zeros(self.n + 1, dtype=np.float64)
        self.m2 = np.zeros(self.n + 1, dtype=np.float64)

        count = 0
        mean = 0.0
        m2 = 0.0
        for index, value in enumerate(array, start=1):
            count += 1
            delta = float(value) - mean
            mean += delta / count
            delta2 = float(value) - mean
            m2 += delta * delta2
            self.count[index] = count
            self.mean[index] = mean
            self.m2[index] = m2

    def _check_bounds(self, start: int, end: int) -> None:
        if not 0 <= start < end <= self.n:
            raise InvalidParameterError(
                "segment bounds must satisfy 0 <= start < end <= n",
                details={"start": start, "end": end, "n": self.n},
            )

    def segment_mean(self, start: int, end: int) -> float:
        self._check_bounds(start, end)
        n_a = int(self.count[start])
        n_b_total = int(self.count[end])
        n_b = n_b_total - n_a
        mean_b = (self.mean[end] * n_b_total - self.mean[start] * n_a) / n_b
        if not math.isfinite(mean_b):
            raise NumericStabilityError(
                "segment mean became non-finite",
                details={"start": start, "end": end},
            )
        return float(mean_b)

    def segment_sse(self, start: int, end: int) -> float:
        """Within-segment sum of squared errors."""
        self._check_bounds(start, end)
        n_a = int(self.count[start])
        n_b_total = int(self.count[end])
        n_b = n_b_total - n_a
        mean_a = float(self.mean[start])
        mean_b = self.segment_mean(start, end)
        delta = mean_b - mean_a
        between = n_a * n_b / n_b_total * delta * delta
        sse = float(self.m2[end] - self.m2[start] - between)

        tolerance = 1e-11 * max(
            1.0,
            abs(float(self.m2[end])),
            abs(float(self.m2[start])),
            abs(between),
        )
        if sse < 0.0:
            if abs(sse) <= tolerance:
                return 0.0
            raise NumericStabilityError(
                "segment SSE was too negative to be explained by floating-point rounding",
                details={"start": start, "end": end, "sse": sse, "tolerance": tolerance},
            )
        return sse
