from __future__ import annotations

import numpy as np
import pytest

from stl_backend.config import StlConfig
from stl_backend.errors import CalculationError, InputContractError, ResourceExhaustedError
from stl_backend.kernel import loess_smooth, robust_bisquare_weights, stl_decompose


CONFIG = StlConfig()


def test_fixed_configuration_contract():
    assert CONFIG.period == 7
    assert CONFIG.seasonal_window == 7
    assert CONFIG.trend_window == 15
    assert CONFIG.low_pass_window == 9
    assert (
        CONFIG.seasonal_degree
        == CONFIG.trend_degree
        == CONFIG.low_pass_degree
        == 1
    )
    assert CONFIG.inner_iterations == 2
    assert CONFIG.outer_iterations == 15
    assert CONFIG.minimum_observations == 14


def test_robust_decomposition_matches_independent_reference(robust_fixture):
    result = stl_decompose(robust_fixture["observed"], CONFIG)
    reference = robust_fixture["reference"]

    np.testing.assert_allclose(
        result["trend"], reference["trend"], rtol=0.0, atol=5e-11
    )
    np.testing.assert_allclose(
        result["seasonal"], reference["seasonal"], rtol=0.0, atol=5e-11
    )
    np.testing.assert_allclose(
        result["residual"], reference["residual"], rtol=0.0, atol=5e-11
    )
    np.testing.assert_allclose(
        result["robust_weights"],
        reference["robust_weights"],
        rtol=0.0,
        atol=1e-12,
    )


def test_seasonal_centering_identity_and_reconstruction(robust_fixture):
    observed = np.asarray(robust_fixture["observed"], dtype=np.float64)
    result = stl_decompose(observed, CONFIG)

    assert abs(float(np.mean(result["seasonal"]))) < 1e-14
    reconstructed = (
        result["trend"] + result["seasonal"] + result["residual"]
    )
    np.testing.assert_allclose(reconstructed, observed, rtol=0.0, atol=1e-12)


def test_robust_weights_downweight_known_outliers(robust_fixture):
    result = stl_decompose(robust_fixture["observed"], CONFIG)
    weights = result["robust_weights"]

    for position in robust_fixture["outlier_positions"]:
        assert weights[position] < 1e-8
    clean_positions = np.array(
        [
            index
            for index in range(len(weights))
            if index not in robust_fixture["outlier_positions"]
        ]
    )
    assert np.mean(weights[clean_positions] > 0.9) > 0.55
    assert np.mean(weights[clean_positions]) > 0.75


def test_known_trend_and_seasonal_shape_are_recovered(robust_fixture):
    observed = np.asarray(robust_fixture["observed"])
    result = stl_decompose(observed, CONFIG)

    known_trend = np.asarray(robust_fixture["known_trend"])
    known_seasonal = np.asarray(robust_fixture["known_seasonal"])
    # Loess estimates are not required to equal the noiseless generating line,
    # but these tolerances distinguish a correct fit from a shifted/windowed one.
    np.testing.assert_allclose(result["trend"], known_trend, atol=0.18)
    np.testing.assert_allclose(result["seasonal"], known_seasonal, atol=0.16)
    assert np.corrcoef(result["trend"], known_trend)[0, 1] > 0.999
    assert np.corrcoef(result["seasonal"], known_seasonal)[0, 1] > 0.998


def test_boundary_loess_uses_local_linear_extrapolation_without_padding():
    y = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    smoothed = loess_smooth(y, window=5, degree=1)

    # Exact local linear fit has no endpoint shrinkage or reflection artifact.
    np.testing.assert_allclose(smoothed, y, atol=1e-12)


def test_seasonal_extension_has_one_period_on_each_side(robust_fixture):
    # Shortest accepted series gives each phase two observations; STL must
    # still extrapolate one value before and after every sub-series.
    observed = np.asarray(robust_fixture["observed"][:CONFIG.minimum_observations])
    result = stl_decompose(observed, CONFIG)

    assert result["trend"].shape == (14,)
    assert result["seasonal"].shape == (14,)
    assert result["residual"].shape == (14,)
    assert abs(float(np.mean(result["seasonal"]))) < 1e-14
    np.testing.assert_allclose(
        result["trend"] + result["seasonal"] + result["residual"],
        observed,
        atol=1e-12,
    )


def test_bisquare_weights_use_stl_scale():
    residual = np.array([0.0, 1.0, -1.0, 10.0])
    weights, cutoff = robust_bisquare_weights(residual)

    assert weights[0] == 1.0
    assert weights[1] == pytest.approx((1.0 - (1.0 / cutoff) ** 2) ** 2)
    assert weights[3] == 0.0
    assert cutoff == pytest.approx(6.0)


def test_missing_and_short_series_are_distinct_input_errors():
    with pytest.raises(InputContractError) as missing:
        stl_decompose([1.0, np.nan] + [1.0] * 12, CONFIG)
    assert "missing" in str(missing.value)

    with pytest.raises(InputContractError) as short:
        stl_decompose([1.0] * 13, CONFIG)
    assert short.value.details["minimum_observations"] == 14
    assert "two complete periods" in str(short.value)


def test_resource_limit_has_its_own_error_category():
    config = StlConfig(max_observations=20)
    with pytest.raises(ResourceExhaustedError) as exc:
        stl_decompose(np.arange(21, dtype=float), config)
    assert exc.value.code == "resource_exhausted"
    assert exc.value.details["maximum_observations"] == 20


def test_calculation_failure_has_distinct_code(monkeypatch):
    from stl_backend import kernel

    def broken_loess(*args, **kwargs):
        raise FloatingPointError("simulated singular local fit")

    monkeypatch.setattr(kernel, "loess_smooth", broken_loess)
    with pytest.raises(CalculationError) as exc:
        stl_decompose(np.arange(14, dtype=float), CONFIG)
    assert exc.value.code == "calculation_failed"
