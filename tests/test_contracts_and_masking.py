from __future__ import annotations

import numpy as np
import pytest

from survey_imputation.contracts import AnalysisRequest, SurveyRecord
from survey_imputation.errors import (
    EmptyDatasetError,
    MaskObservedMismatchError,
    MaskShapeError,
)
from survey_imputation.masking import ensure_outcome_column, missing_mask


def test_zero_is_observed_and_nan_is_masked():
    y = np.array([0.0, np.nan, 2.5], dtype=float)

    mask = missing_mask(y)

    assert mask.tolist() == [False, True, False]


def test_mask_rejects_true_value_marked_missing_including_zero():
    y = np.array([0.0, np.nan], dtype=float)

    with pytest.raises(MaskObservedMismatchError) as exc_info:
        missing_mask(y, np.array([True, True]))

    assert exc_info.value.category == "mask_observed_mismatch"
    assert "zeros are observed" in str(exc_info.value)


def test_mask_rejects_missing_value_marked_observed():
    y = np.array([1.0, np.nan], dtype=float)

    with pytest.raises(MaskObservedMismatchError) as exc_info:
        missing_mask(y, np.array([False, False]))

    assert exc_info.value.category == "mask_observed_mismatch"


def test_mask_shape_and_empty_input_have_specific_categories():
    with pytest.raises(MaskShapeError) as shape_error:
        missing_mask(np.array([1.0, np.nan]), np.array([False]))
    assert shape_error.value.category == "mask_shape_mismatch"

    with pytest.raises(EmptyDatasetError) as empty_error:
        missing_mask(np.array([], dtype=float))
    assert empty_error.value.category == "empty_dataset"


def test_contract_preserves_none_and_accepts_real_zero():
    record = SurveyRecord(
        record_id="zero",
        age_z=0.0,
        member=0,
        used_support=0,
        contact_frequency=0.0,
        rating=0,
    )
    assert record.rating == 0.0
    outcome = ensure_outcome_column([0.0, None])
    assert outcome[0] == 0.0
    assert np.isnan(outcome[1])


def test_analysis_contract_requires_records():
    with pytest.raises(ValueError):
        AnalysisRequest(records=[], method="mean_imputation")
