from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from survey_backend.errors import DuplicateRecordIdError, EmptyDatasetError, IneligibleObservationError
from survey_backend.models import SurveyRecord

FEATURE_NAMES = ("q1_screen_hours", "q2_sleep_hours")
OUTCOME_NAME = "satisfaction"
MODEL_COLUMNS = (*FEATURE_NAMES, OUTCOME_NAME)
COEFFICIENT_NAMES = ("q1_screen_hours", "q2_sleep_hours", "intercept")


@dataclass(frozen=True)
class AnalysisData:
    record_ids: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    missing: np.ndarray
    eligible: np.ndarray

    @property
    def n(self) -> int:
        return self.x.shape[0]

    def allowed_indices(self) -> np.ndarray:
        return np.flatnonzero(self.eligible)

    def missing_counts_allowed(self) -> dict[str, int]:
        allowed = self.eligible
        return {
            "q1_screen_hours": int(np.sum(self.missing[:, 0][allowed])),
            "q2_sleep_hours": int(np.sum(self.missing[:, 1][allowed])),
            "satisfaction": int(np.sum(self.missing[:, 2][allowed])),
        }

    def zero_counts_allowed(self) -> dict[str, int]:
        values = np.column_stack([self.x, self.y])
        allowed = self.eligible
        observed = ~self.missing
        return {
            column: int(np.sum((values[:, i] == 0.0) & observed[:, i] & allowed))
            for i, column in enumerate(MODEL_COLUMNS)
        }


def prepare_records(records: list[SurveyRecord]) -> AnalysisData:
    if not records:
        raise EmptyDatasetError("records must contain at least one participant")

    ids = tuple(record.record_id for record in records)
    if len(set(ids)) != len(ids):
        raise DuplicateRecordIdError("record_id must be unique within a request")

    values = np.full((len(records), 3), np.nan, dtype=float)
    missing = np.zeros((len(records), 3), dtype=bool)
    eligible = np.zeros(len(records), dtype=bool)
    raw_values = [
        (record.q1_screen_hours, record.q2_sleep_hours, record.satisfaction)
        for record in records
    ]
    for i, (q1, q2, y) in enumerate(raw_values):
        for j, value in enumerate((q1, q2, y)):
            if value is None:
                missing[i, j] = True
            else:
                values[i, j] = float(value)
        eligible[i] = records[i].eligible

    data = AnalysisData(
        record_ids=ids,
        x=values[:, :2].copy(),
        y=values[:, 2].copy(),
        missing=missing,
        eligible=eligible,
    )
    if not np.any(eligible):
        raise IneligibleObservationError("at least one eligible observation is required")
    return data


def observed_design_mask(data: AnalysisData, include_outcome: bool = True) -> np.ndarray:
    feature_observed = ~np.any(data.missing[:, :2], axis=1)
    if include_outcome:
        return data.eligible & feature_observed & ~data.missing[:, 2]
    return data.eligible & feature_observed


def design_matrix(x: np.ndarray) -> np.ndarray:
    return np.column_stack([x, np.ones(x.shape[0], dtype=float)])
