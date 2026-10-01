"""Sample contract, fixed preprocessing, and result data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from app.errors import SampleContractError


class MethodName(StrEnum):
    EMPIRICAL = "empirical"
    LEDOIT_WOLF = "ledoit_wolf"
    OGK = "ogk"


@dataclass(frozen=True)
class PreparedSample:
    """Data ready for an estimator, with the fixed preprocessing audit trail."""

    values: np.ndarray
    n_samples: int
    n_features: int
    feature_names: list[str]
    missing_cells: int
    imputed_with: list[float]
    robust_center: list[float]
    robust_scale: list[float]
    scale_policy: str
    missing_policy: str
    warnings: list[str] = field(default_factory=list)

    def as_report(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "missing": {
                "policy": self.missing_policy,
                "missing_cells": self.missing_cells,
                "imputed_with": self.imputed_with,
            },
            "scaling": {
                "policy": self.scale_policy,
                "robust_center": self.robust_center,
                "robust_scale": self.robust_scale,
            },
            "warnings": self.warnings,
        }


def prepare_sample(
    rows: list[list[float | None]],
    feature_names: list[str] | None = None,
) -> PreparedSample:
    """Validate shape and apply the only supported missing-value policy.

    Missing cells are replaced by their per-column median after rejecting
    non-finite values and columns that have no observed cell.  Scaling itself
    is estimator-specific; this function always reports the fixed MAD scale so
    requests can be audited and tests can verify it.
    """

    if not isinstance(rows, list) or not rows:
        raise SampleContractError(
            "At least two samples are required.",
            stage="sample_contract",
            details={"n_samples": len(rows) if isinstance(rows, list) else 0},
        )

    n_features = len(rows[0]) if isinstance(rows[0], list) else -1
    if len(rows) < 2:
        raise SampleContractError(
            "At least two samples are required.",
            stage="sample_contract",
            details={"n_samples": len(rows)},
        )
    if n_features < 1:
        raise SampleContractError(
            "At least one feature column is required.",
            stage="sample_contract",
            details={"n_features": n_features},
        )

    for row_index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != n_features:
            raise SampleContractError(
                "Every row must be a list with the same number of columns.",
                stage="sample_contract",
                details={"row_index": row_index, "expected_columns": n_features},
            )

    raw = np.array([[np.nan if value is None else value for value in row] for row in rows], dtype=float)
    observed_mask = np.array([[value is not None for value in row] for row in rows], dtype=bool)
    if raw.shape != (len(rows), n_features):
        raise SampleContractError(
            "The sample matrix could not be represented as a floating-point matrix.",
            stage="sample_contract",
            details={"shape": list(raw.shape)},
        )

    missing_mask = np.isnan(raw)
    invalid_cells = np.argwhere(observed_mask & ~np.isfinite(raw))
    if invalid_cells.size:
        positions = [[int(i), int(j)] for i, j in invalid_cells[:10]]
        raise SampleContractError(
            "Only null values may represent missing observations; NaN and infinity are invalid.",
            stage="sample_contract",
            details={"invalid_positions": positions},
        )

    observed_count = np.count_nonzero(~missing_mask, axis=0)
    empty_columns = np.flatnonzero(observed_count == 0).tolist()
    if empty_columns:
        raise SampleContractError(
            "Each column needs at least one observed value for median imputation.",
            stage="missing_handling",
            details={"empty_columns": [int(j) for j in empty_columns]},
        )

    values = raw.copy()
    imputation: list[float] = []
    warnings: list[str] = []
    for j in range(n_features):
        observed = raw[~missing_mask[:, j], j]
        median = float(np.median(observed))
        imputation.append(median)
        values[missing_mask[:, j], j] = median
        if missing_mask[:, j].any():
            warnings.append(f"Feature {j} had missing observations replaced by its observed median.")

    if feature_names is None:
        names = [f"x{j}" for j in range(n_features)]
    else:
        if len(feature_names) != n_features:
            raise SampleContractError(
                "feature_names length must match the number of sample columns.",
                stage="sample_contract",
                details={"expected": n_features, "received": len(feature_names)},
            )
        names = [str(name) for name in feature_names]

    center, scale = robust_location_scale(values)
    missing_cells = int(missing_mask.sum())
    return PreparedSample(
        values=values,
        n_samples=len(rows),
        n_features=n_features,
        feature_names=names,
        missing_cells=missing_cells,
        imputed_with=imputation,
        robust_center=center.tolist(),
        robust_scale=scale.tolist(),
        scale_policy="Per-column median location and Gaussian-consistent MAD scale are reported; estimators retain or apply and reverse this fixed scale explicitly.",
        missing_policy="Null cells are imputed with the median of observed values in the same column.",
        warnings=warnings,
    )


def robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return median and MAD with the Gaussian consistency constant."""

    center = np.median(values, axis=0)
    raw_mad = np.median(np.abs(values - center), axis=0)
    scale = raw_mad / 0.6744897501960817
    return center, scale
