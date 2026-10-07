"""Validation and duplicate-score merging rules."""
import numpy as np
import pytest

from calibration_service.app.validation import (
    ValidationError,
    merge_duplicate_scores,
    validate_arrays,
)


def test_zero_weight_rows_are_dropped_and_counted():
    s, y, w, summary = validate_arrays(
        np.array([0.1, 0.2, 0.3]),
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, 0.0, 2.0]),
    )
    assert summary.n_zero_weight_dropped == 1
    assert summary.n_used == 2
    np.testing.assert_allclose(s, [0.1, 0.3])
    np.testing.assert_allclose(w, [1.0, 2.0])


def test_all_zero_weights_is_an_error():
    with pytest.raises(ValidationError, match="zero weight"):
        validate_arrays(np.array([0.1]), np.array([1.0]), np.array([0.0]))


def test_nan_score_rejected_with_row_index():
    with pytest.raises(ValidationError) as exc:
        validate_arrays(
            np.array([0.1, np.nan]), np.array([0.0, 1.0]), np.array([1.0, 1.0])
        )
    assert exc.value.row_indices == [1]


def test_negative_weight_rejected():
    with pytest.raises(ValidationError, match="negative"):
        validate_arrays(
            np.array([0.1, 0.2]), np.array([0.0, 1.0]), np.array([1.0, -0.5])
        )


def test_label_outside_zero_one_rejected():
    with pytest.raises(ValidationError, match="label"):
        validate_arrays(
            np.array([0.1, 0.2]), np.array([0.0, 0.5]), np.array([1.0, 1.0])
        )


def test_length_mismatch_rejected():
    with pytest.raises(ValidationError, match="equal length"):
        validate_arrays(np.array([0.1]), np.array([0.0, 1.0]), np.array([1.0]))


def test_duplicate_scores_merge_by_weight():
    # Two rows at score 0.5: (label 1, w 1) and (label 0, w 3)
    # merge into label 0.25, weight 4.
    s, y, w, n_merged = merge_duplicate_scores(
        np.array([0.5, 0.1, 0.5]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    assert n_merged == 1
    np.testing.assert_allclose(s, [0.1, 0.5])
    np.testing.assert_allclose(y, [0.0, 0.25])
    np.testing.assert_allclose(w, [2.0, 4.0])


def test_merge_returns_sorted_output():
    s, y, w, n_merged = merge_duplicate_scores(
        np.array([0.9, 0.1, 0.5]),
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
    )
    assert n_merged == 0
    np.testing.assert_allclose(s, [0.1, 0.5, 0.9])
