from __future__ import annotations

import numpy as np

from app.sample_contract import SampleContractError, prepare_sample


def test_fixed_missing_imputation_location_and_scale() -> None:
    data = [[1.0, 100.0], [3.0, None], [5.0, 300.0], [7.0, 400.0]]
    prepared = prepare_sample(data, assume_centered=False, center_method="mean")

    assert prepared.mask.tolist() == [
        [False, False],
        [False, True],
        [False, False],
        [False, False],
    ]
    assert prepared.imputed[1, 1] == 350.0
    assert prepared.complete_rows == 3
    np.testing.assert_allclose(prepared.center, [4.0, 287.5])
    # Constant columns are not silently expanded; supplied scales are honored.
    fixed = prepare_sample(
        [[2.0], [2.0], [2.0]],
        assume_centered=False,
        supplied_scales=[7.0],
    )
    assert fixed.scales.tolist() == [7.0]


def test_empty_column_has_named_failure_category() -> None:
    try:
        prepare_sample([[None, 1.0], [None, 2.0]], assume_centered=False)
    except SampleContractError as exc:
        assert str(exc).startswith("EMPTY_COLUMN:")
    else:
        raise AssertionError("empty column must be rejected")
