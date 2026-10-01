from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from survey_backend.statistical.data import AnalysisData
from survey_backend.statistical.validation import (
    observed_predictor_counts,
    observed_outcome_rows,
    reject_rows_with_all_predictors_missing,
)


@dataclass(frozen=True)
class CompletedData:
    x: np.ndarray
    y: np.ndarray
    record_indices: np.ndarray
    imputed_counts: tuple[int, int]
    is_imputed: np.ndarray
    stream_tag: str = ""


def mean_impute_predictors(data: AnalysisData) -> CompletedData:
    rows = observed_outcome_rows(data)
    observed_predictor_counts(data, rows)
    reject_rows_with_all_predictors_missing(data, rows)

    x = data.x[rows, :].copy()
    y = data.y[rows].copy()
    is_imputed = data.missing[rows, :2].copy()
    for j in range(2):
        observed = ~data.missing[rows, j]
        x[~observed, j] = float(np.mean(x[observed, j]))
    return CompletedData(
        x=x,
        y=y,
        record_indices=rows,
        imputed_counts=(int(np.sum(is_imputed[:, 0])), int(np.sum(is_imputed[:, 1]))),
        is_imputed=is_imputed,
    )
