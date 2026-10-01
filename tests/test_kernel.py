from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from app.config import FIXED_CONFIG, PERIOD
from app.errors import ComputationError, InputError, ResourceExhaustedError
from app.kernel.filters import moving_average
from app.kernel.loess import loess_fit_at
from app.kernel.stl import decompose_stl


def sha256_json(values: np.ndarray) -> str:
    encoded = json.dumps(
        np.asarray(values, dtype=np.float64).tolist(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_primary_fixture_matches_mature_statsmodels_reference(primary_fixture):
    reference = primary_fixture["statsmodels_reference"]
    result = decompose_stl(primary_fixture["values"])

    for component in ("trend", "seasonal", "residual", "weights"):
        actual = getattr(result, component)
        expected = np.asarray(reference[component], dtype=np.float64)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=2e-12,
            err_msg=f"{component} differs from independent statsmodels reference",
        )


def test_primary_fixture_locks_specific_versioned_results(primary_fixture):
    result = decompose_stl(primary_fixture["values"])
    expected_hashes = {
        "trend": "c6bc86e9ee40ace8c0bf23afe8eb9eb2b95c97ee061d2bc9bf71e8fcb9a29c30",
        "seasonal": "5bc61566b52655176756e6f26533b8bbacded0fe5673b105b99a0a590cf272cd",
        "residual": "5ff475dd936c54db54495711ee149e7e49869bd3ac4d9ff2b0c84937dda6d6ce",
        "weights": "eee9b29d5352687ce5fd6729ca27f99e810e44da0a9d4be3b1a5d4f6c4f429c2",
    }
    for component, expected_hash in expected_hashes.items():
        assert sha256_json(getattr(result, component)) == expected_hash


def test_contracts_center_seasonal_and_reconstruct_additively(primary_fixture):
    observed = np.asarray(primary_fixture["values"], dtype=np.float64)
    result = decompose_stl(observed)

    assert abs(np.mean(result.seasonal)) < 1e-14
    np.testing.assert_allclose(
        observed,
        result.trend + result.seasonal + result.residual,
        rtol=0.0,
        atol=1e-12,
    )
    assert result.diagnostics["additive_max_abs_error"] == pytest.approx(0.0, abs=1e-12)
    assert result.diagnostics["config_fingerprint"]
    assert result.diagnostics["fixed_config"] == FIXED_CONFIG


def test_robust_weights_specifically_downweight_known_outliers(primary_fixture):
    result = decompose_stl(primary_fixture["values"])
    outlier_indices = primary_fixture["outlier_indices"]

    assert result.weights[outlier_indices].tolist() == [0.0, 0.0, 0.0]
    clean_check_indices = [0, 1, 5, 10, 20, 30, 40, 50, 65]
    assert np.min(result.weights[clean_check_indices]) > 0.55
    assert result.diagnostics["zero_weight_count"] >= 3

    last_trace = result.trace[-1]
    assert last_trace["stage"] == "robust_weight_update"
    assert last_trace["judgment"]["zero_weight_count"] >= 3
    assert last_trace["messages"] == [
        "biweight_weights_updated_from_residual_median_absolute_deviation"
    ]


def test_boundary_minimum_length_uses_extensions_and_matches_reference(
    boundary_fixture,
):
    reference = boundary_fixture["statsmodels_reference"]
    result = decompose_stl(boundary_fixture["values"])

    assert result.diagnostics["observations"] == 2 * PERIOD
    np.testing.assert_allclose(
        result.trend,
        reference["trend"],
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result.seasonal,
        reference["seasonal"],
        rtol=0.0,
        atol=2e-12,
    )
    assert result.trend[0] == pytest.approx(0.013847024220263595, abs=1e-12)
    assert result.trend[-1] == pytest.approx(1.285296231044462, abs=1e-12)


def test_missing_values_are_rejected_and_not_zero_filled():
    values = np.arange(14, dtype=np.float64)
    values[3] = np.nan
    with pytest.raises(InputError) as exc_info:
        decompose_stl(values)
    assert exc_info.value.error_type == "INPUT_ERROR"
    assert exc_info.value.details["non_finite_indices"] == [3]
    assert exc_info.value.details["non_finite_count"] == 1


def test_short_series_reports_exact_required_periods():
    with pytest.raises(InputError) as exc_info:
        decompose_stl([1.0] * 13)
    assert exc_info.value.error_type == "INPUT_ERROR"
    assert exc_info.value.details == {
        "received_observations": 13,
        "required_observations": 14,
        "period": 7,
    }


def test_too_large_series_is_distinct_resource_error():
    with pytest.raises(ResourceExhaustedError) as exc_info:
        decompose_stl([1.0] * 5001)
    assert exc_info.value.error_type == "RESOURCE_EXHAUSTED"
    assert exc_info.value.details["maximum_observations"] == 5000


def test_local_linear_boundary_extrapolation_exact_for_straight_line():
    x = np.array([2.0, 4.0, 6.0, 8.0])
    left = loess_fit_at(x, target=-1, length=3, degree=1)
    right = loess_fit_at(x, target=4, length=3, degree=1)
    assert left.value == pytest.approx(0.0, abs=1e-12)
    assert right.value == pytest.approx(10.0, abs=1e-12)


def test_invalid_moving_average_is_computation_failure():
    with pytest.raises(ComputationError) as exc_info:
        moving_average(np.ones(3), 5)
    assert exc_info.value.error_type == "COMPUTATION_FAILED"
