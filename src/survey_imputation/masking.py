"""Missing mask construction and validation.

The project's invariant is deliberately strict:

* ``NaN``/``None`` in the outcome means missing.
* ``mask == 1`` means missing.
* Exact zero is observed and must have ``mask == 0``.

This prevents a common data-preparation bug in which legitimate zero ratings
are silently treated as missing.
"""

from __future__ import annotations

import numpy as np

from .errors import (
    EmptyDatasetError,
    MaskObservedMismatchError,
    MaskShapeError,
    MissingOutcomeError,
)


def missing_mask(y: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise EmptyDatasetError("outcome vector must contain at least one row")

    derived = np.isnan(y)
    if mask is None:
        return derived.astype(bool)

    provided = np.asarray(mask)
    if provided.shape != y.shape:
        raise MaskShapeError(f"mask shape {provided.shape} does not equal outcome shape {y.shape}")
    try:
        provided_bool = provided.astype(bool, copy=True)
    except (TypeError, ValueError) as exc:
        raise MaskObservedMismatchError("mask must contain boolean-compatible values") from exc

    marked_but_value_present = provided_bool & ~derived
    if np.any(marked_but_value_present):
        index = int(np.flatnonzero(marked_but_value_present)[0])
        raise MaskObservedMismatchError(
            f"row {index} is marked missing but has a non-missing value; zeros are observed"
        )

    value_missing_but_unmarked = derived & ~provided_bool
    if np.any(value_missing_but_unmarked):
        index = int(np.flatnonzero(value_missing_but_unmarked)[0])
        raise MaskObservedMismatchError(f"row {index} has missing outcome but mask marks it observed")

    return provided_bool


def ensure_outcome_column(values: list[float | None]) -> np.ndarray:
    if not values:
        raise EmptyDatasetError("records must contain at least one row")
    try:
        return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)
    except (TypeError, ValueError) as exc:
        raise MissingOutcomeError("rating must be numeric or null") from exc
