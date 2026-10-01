from __future__ import annotations

import math

import arviz as az
import numpy as np
import pytest
from statsmodels.tsa.stattools import acf as statsmodels_acf

from app.mcmc_diagnostics.config import get_settings
from app.mcmc_diagnostics.kernel import analyze_chains, autocovariance_chain
from app.mcmc_diagnostics.models import MultiChainRequest


def test_acf_uses_centering_biased_covariance_and_matches_statsmodels():
    rng = np.random.default_rng(42)
    x = rng.normal(size=500)
    ours = autocovariance_chain(x)
    ours = ours[:41] / ours[0]
    external = statsmodels_acf(x, nlags=40, fft=True, missing="none")

    n = x.size
    centered = x - x.mean()
    expected_lag_1 = float(np.dot(centered[:-1], centered[1:]) / n / (np.dot(centered, centered) / n))

    assert ours[0] == pytest.approx(1.0)
    assert ours[1] == pytest.approx(expected_lag_1, rel=1e-12, abs=1e-12)
    assert ours == pytest.approx(external, abs=1e-12)


def test_independent_noise_does_not_collapse_or_inflate_ess():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(4, 5_000))
    result = analyze_chains(MultiChainRequest(chains=data.tolist()), get_settings())
    by_kind = {item.kind: item for item in result.ess}

    assert by_kind["split_mean"].ess == pytest.approx(
        float(az.ess(data, method="mean")), rel=0.02
    )
    assert by_kind["split_mean"].relative_ess > 0.90
    rank_rhat = next(item for item in result.rhat if item.kind == "rank_normalized")
    assert 0.98 < rank_rhat.rhat < 1.01
    assert result.status == "ok"


def test_ar1_ess_is_degraded_and_matches_external_and_theory(ar1_data):
    result = analyze_chains(MultiChainRequest(chains=ar1_data.tolist()), get_settings())
    by_kind = {item.kind: item for item in result.ess}

    phi = 0.8
    theoretical_iat = (1 + phi) / (1 - phi)
    theoretical_ess = ar1_data.size / theoretical_iat
    external = float(az.ess(ar1_data, method="mean"))

    assert by_kind["split_mean"].ess == pytest.approx(external, rel=0.03)
    assert by_kind["split_mean"].ess == pytest.approx(theoretical_ess, rel=0.10)
    assert by_kind["split_mean"].integrated_time == pytest.approx(theoretical_iat, rel=0.10)
    assert by_kind["split_mean"].ess < 0.30 * ar1_data.size
    assert by_kind["split_mean"].truncation_rule.startswith("scan complete lag pairs")


def test_ar1_acf_decays_to_asymptotic_ar1_shape(ar1_data):
    result = analyze_chains(MultiChainRequest(chains=ar1_data.tolist()), get_settings())
    chain_acf = result.acf[0].acf
    phi = 0.8
    for lag in (1, 5, 10):
        assert chain_acf[lag] == pytest.approx(phi**lag, abs=0.06)
    assert result.acf[0].method == "per_chain_centered_biased_autocovariance_divided_by_lag_zero"


def test_shifted_chain_means_separate_within_and_between_variance(shifted_data):
    result = analyze_chains(MultiChainRequest(chains=shifted_data.tolist()), get_settings())
    components = result.variance_components
    n = shifted_data.shape[1]
    means = shifted_data.mean(axis=1)
    expected_within = float(shifted_data.var(axis=1, ddof=1).mean())
    expected_between = float(n * np.var(means, ddof=1))
    expected_classic_rhat = math.sqrt(
        ((n - 1) / n * expected_within + expected_between / n) / expected_within
    )

    assert components.within_chain_variance == pytest.approx(expected_within, abs=1e-10)
    assert components.between_chain_variance == pytest.approx(expected_between, rel=1e-10)
    assert components.marginal_variance == pytest.approx(
        (n - 1) / n * expected_within + expected_between / n, rel=1e-10
    )
    rhat = {item.kind: item for item in result.rhat}
    assert rhat["classic"].rhat == pytest.approx(expected_classic_rhat, rel=1e-10)
    assert rhat["classic"].rhat == pytest.approx(float(az.rhat(shifted_data, method="identity")), rel=1e-10)
    assert rhat["rank_normalized"].rhat == pytest.approx(
        float(az.rhat(shifted_data, method="rank")), rel=1e-8
    )
    assert rhat["classic"].rhat > 2.0
    assert "RHAT_ABOVE_THRESHOLD" in {d.code for d in result.diagnostics}
    ess = {item.kind: item for item in result.ess}
    assert ess["split_mean"].ess < 30


def test_constant_chains_return_undefined_not_n_independent_draws():
    data = np.array([[7.0] * 20, [7.0] * 20, [9.0] * 20, [9.0] * 20])
    result = analyze_chains(MultiChainRequest(chains=data.tolist()), get_settings())

    assert result.status == "inconclusive"
    assert {item.state for item in result.ess} == {"not_estimated"}
    assert all(item.rhat is None for item in result.rhat)
    assert {item.state for item in result.rhat} == {"undefined"}
    codes = {d.code for d in result.diagnostics}
    assert {"CONSTANT_CHAIN", "RHAT_ZERO_WITHIN_VARIANCE"}.issubset(codes)
    assert len(result.acf[0].acf) == 20
    assert all(value is None for value in result.acf[0].acf)


def test_global_constant_single_chain_has_specific_category():
    result = analyze_chains(
        MultiChainRequest(chains=[[2.0] * 30, [2.0] * 30]), get_settings()
    )
    assert result.status == "inconclusive"
    assert {d.code for d in result.diagnostics} >= {"CONSTANT_CHAIN", "RHAT_ZERO_WITHIN_VARIANCE"}
    assert all(item.ess is None for item in result.ess)


@pytest.mark.parametrize("draws", [1, 3, 5, 9])
def test_short_chains_do_not_return_silently_wrong_ess(draws):
    rng = np.random.default_rng(100 + draws)
    data = rng.normal(size=(2, draws))
    result = analyze_chains(MultiChainRequest(chains=data.tolist()), get_settings())

    assert "SHORT_CHAIN" in {d.code for d in result.diagnostics}
    assert all(item.ess is None for item in result.ess)
    if draws < 4:
        assert {item.state for item in result.rhat} == {"not_estimated"}
    else:
        assert any(item.state == "estimated" for item in result.rhat)


def test_unequal_length_keeps_variance_components_without_fake_aggregate():
    request = MultiChainRequest(
        chains=[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0]],
        require_equal_length=False,
    )
    result = analyze_chains(request, get_settings())
    assert result.variance_components.equal_length is False
    assert result.variance_components.within_chain_variance == pytest.approx(1.4)
    assert result.variance_components.between_chain_variance == pytest.approx(
        -0.2833333333333333
    )
    assert all(item.rhat is None for item in result.rhat)
    assert "UNEQUAL_LENGTH_INPUT" in {d.code for d in result.diagnostics}


def test_default_display_lag_is_separate_from_ess_scan():
    rng = np.random.default_rng(9)
    data = rng.normal(size=(4, 10_000))
    result = analyze_chains(MultiChainRequest(chains=data.tolist()), get_settings())
    assert result.acf[0].max_lag == 40
    ess = next(item for item in result.ess if item.kind == "split_mean")
    assert ess.cutoff_lag <= 10
    assert ess.total_draws == 40_000
