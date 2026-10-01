from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SampleContractError(ValueError):
    """Failure category for invalid or unusable sample matrices."""


@dataclass(frozen=True)
class PreparedSample:
    raw: np.ndarray
    mask: np.ndarray
    imputed: np.ndarray
    center: np.ndarray
    residuals: np.ndarray
    scales: np.ndarray
    standardized: np.ndarray
    missing_count: int
    complete_rows: int


def prepare_sample(
    data: list[list[float | None]],
    *,
    assume_centered: bool,
    center_method: str = "mean",
    supplied_scales: list[float] | None = None,
) -> PreparedSample:
    """Apply the fixed imputation, location, and column-scale contract.

    Missing entries are median-imputed independently by column. Location is
    the arithmetic mean of the imputed matrix unless ``assume_centered`` is
    true. Scales are the supplied values or MAD normalized for a Gaussian.
    Covariances are estimated on standardized residuals and mapped back to
    the original units by diag(scales) * C * diag(scales).
    """
    raw = np.array(data, dtype=float)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise SampleContractError("DATA_SHAPE: data must be a nonempty 2-D matrix")

    mask = np.isnan(raw)
    imputed = raw.copy()
    for j in range(raw.shape[1]):
        observed = raw[~mask[:, j], j]
        if observed.size == 0:
            raise SampleContractError(
                f"EMPTY_COLUMN: column {j} contains no observed values"
            )
        imputed[mask[:, j], j] = np.median(observed)

    if not np.all(np.isfinite(imputed)):
        raise SampleContractError("NON_FINITE: all values must be finite or null")

    if assume_centered:
        center = np.zeros(raw.shape[1], dtype=float)
    elif center_method == "median":
        center = np.median(imputed, axis=0)
    elif center_method == "mean":
        center = imputed.mean(axis=0)
    else:
        raise SampleContractError(
            f"BAD_CENTER_METHOD: unsupported center method {center_method}"
        )
    residuals = imputed - center

    if supplied_scales is not None:
        scales = np.asarray(supplied_scales, dtype=float)
        if scales.shape != (raw.shape[1],) or np.any(scales <= 0.0):
            raise SampleContractError(
                "BAD_SCALES: column scales must be positive and match width"
            )
    else:
        # 1.4826 makes MAD an unbiased scale estimate at Gaussian columns.
        scales = np.median(np.abs(residuals - np.median(residuals, axis=0)), axis=0)
        scales *= 1.4826
        # Constant columns retain their zero spread instead of being expanded.
        scales[scales <= np.finfo(float).eps] = 1.0

    standardized = residuals / scales[np.newaxis, :]
    complete_rows = int(np.sum(~np.any(mask, axis=1)))
    return PreparedSample(
        raw=raw,
        mask=mask,
        imputed=imputed,
        center=center,
        residuals=residuals,
        scales=scales,
        standardized=standardized,
        missing_count=int(mask.sum()),
        complete_rows=complete_rows,
    )
