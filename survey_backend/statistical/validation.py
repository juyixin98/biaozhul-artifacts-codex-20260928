from __future__ import annotations

import numpy as np

from survey_backend.errors import (
    AllMissingFieldError,
    InsufficientRowsError,
    MissingAllPredictorsError,
)
from survey_backend.statistical.data import AnalysisData


def validate_allowed_fields(data: AnalysisData) -> None:
    allowed = data.eligible
    for index, name in enumerate(("q1_screen_hours", "q2_sleep_hours", "satisfaction")):
        column = np.column_stack([data.x, data.y])[:, index]
        observed = ~np.isnan(column)
        if not np.any(observed & allowed):
            raise AllMissingFieldError(f"{name} is missing for every allowed observation")


def complete_case_rows(data: AnalysisData) -> np.ndarray:
    validate_allowed_fields(data)
    rows = np.flatnonzero(
        data.eligible
        & ~np.any(data.missing, axis=1)
    )
    if rows.size < 4:
        raise InsufficientRowsError(
            f"complete-case analysis requires at least 4 fully observed allowed rows, found {rows.size}"
        )
    return rows


def observed_outcome_rows(data: AnalysisData) -> np.ndarray:
    validate_allowed_fields(data)
    rows = np.flatnonzero(data.eligible & ~data.missing[:, 2])
    if rows.size < 4:
        raise InsufficientRowsError(
            f"mean and multiple imputation require at least 4 allowed observed-outcome rows, found {rows.size}"
        )
    return rows


def observed_predictor_counts(data: AnalysisData, rows: np.ndarray) -> tuple[int, int]:
    q1_n = int(np.sum(~data.missing[rows, 0]))
    q2_n = int(np.sum(~data.missing[rows, 1]))
    if q1_n < 3 or q2_n < 3:
        raise InsufficientRowsError("each predictor needs at least three observed allowed values")
    return q1_n, q2_n


def reject_rows_with_all_predictors_missing(data: AnalysisData, rows: np.ndarray) -> None:
    all_predictors_missing = rows[np.all(data.missing[rows, :2], axis=1)]
    if all_predictors_missing.size:
        record = data.record_ids[int(all_predictors_missing[0])]
        raise MissingAllPredictorsError(
            f"record {record} has both predictors missing; conditional imputation needs at least one predictor"
        )
