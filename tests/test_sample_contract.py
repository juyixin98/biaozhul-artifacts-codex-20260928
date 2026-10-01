from __future__ import annotations

import math

import numpy as np
import pytest

from app.covariance.sample import prepare_sample
from app.errors import SampleContractError
from tests.fixtures import missing_dataset


def test_missing_values_use_fixed_column_median_and_reported_mad_scale() -> None:
    completed, rows, medians = missing_dataset()
    prepared = prepare_sample(rows, ["a", "b", "c", "d"])

    assert np.allclose(prepared.values, completed)
    assert prepared.missing_cells == 2
    assert prepared.imputed_with[0] == pytest.approx(medians[0])
    assert prepared.imputed_with[3] == pytest.approx(medians[1])
    expected_scale = np.median(np.abs(completed - np.median(completed, axis=0)), axis=0) / 0.6744897501960817
    assert np.allclose(prepared.robust_scale, expected_scale)
    assert prepared.feature_names == ["a", "b", "c", "d"]


def test_empty_column_has_missing_handling_failure_category() -> None:
    rows = [[1.0, None], [2.0, None], [3.0, 4.0]]
    with pytest.raises(SampleContractError) as excinfo:
        prepare_sample(rows)
    assert excinfo.value.code == "sample_contract_error"
    assert excinfo.value.stage == "missing_handling"
    assert excinfo.value.details["empty_columns"] == [1]


def test_ragged_rows_fail_shape_contract() -> None:
    with pytest.raises(SampleContractError) as excinfo:
        prepare_sample([[1.0, 2.0], [3.0]])
    assert excinfo.value.stage == "sample_contract"
    assert excinfo.value.details["row_index"] == 1


def test_non_finite_observed_values_are_rejected_even_though_null_is_allowed() -> None:
    rows = [[1.0, 2.0], [float("inf"), 4.0]]
    with pytest.raises(SampleContractError) as excinfo:
        prepare_sample(rows)
    assert excinfo.value.stage == "sample_contract"
    assert excinfo.value.details["invalid_positions"] == [[1, 0]]
    assert math.isfinite(float(rows[0][0]))
