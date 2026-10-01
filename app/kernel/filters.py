"""Moving-average filters used after seasonal subseries smoothing."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.errors import ComputationError


def moving_average(values: NDArray[np.float64], length: int) -> NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64)
    if length <= 0 or length > x.size:
        raise ComputationError(
            "Moving-average window is not valid for available samples",
            {"n": int(x.size), "window": int(length)},
        )
    if length == 1:
        return x.copy()
    kernel = np.full(length, 1.0 / length, dtype=np.float64)
    return np.convolve(x, kernel, mode="valid")

