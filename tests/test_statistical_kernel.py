from __future__ import annotations

import numpy as np
import pytest
import arviz as az
from statsmodels.tsa.stattools import acf as sm_acf

from mcmc_diagnostics.kernel import (
    DiagnosticInputError,
    analyze_chains,
    chain_autocorrelation,
)


def ar1(phi: float, n_chains: int, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.zeros((n_chains, n_draws))
    scale = np.sqrt(1.0 - phi**2)
    for chain in range(n_chains):
        error = rng.normal(scale=scale, size=n_draws)
        output[chain, 0] = error[0]
        for draw in range(1, n_draws):
            output[chain, draw] = phi * output[chain, draw - 1] + error[draw]
    return output


def test_acf_uses_within_chain_demeaning_and_matches_statsmodels():
    rng = np.random.default_rng(7)
    chain = 3.0 + 2.0 * rng.normal(size=500)
    values, not_estimated = chain_autocorrelation(chain, max_lag=30)
    expected = sm_acf(chain, nlags=30, fft=True)
    assert not_estimated is False
    assert values is not None
    assert values[0] == pytest.approx(1.0)
    np.testing.assert_allclose(values, expected, atol=1e-12)


def test_ar1_acf_reflects_dependence_so_samples_are_not_counted_as_independent():
    chain_matrix = ar1(phi=0.8, n_chains=1, n_draws=2_000, seed=11)
    values, _ = chain_autocorrelation(chain_matrix[0], max_lag=5)
    assert values is not None
    assert values[1] == pytest.approx(0.8, abs=0.05)


def test_independent_noise_ess_matches_arviz_and_stays_near_nominal_count():
    rng = np.random.default_rng(21)
    samples = rng.normal(size=(4, 2_000))
    result = analyze_chains(samples, default_max_acf_lag=10)
    ours = result["effective_sample_size"]
    assert ours["identity_original_chains"]["value"] == pytest.approx(
        az.ess(samples, method="identity"), rel=1e-12
    )
    assert ours["mean_split_chains"]["value"] == pytest.approx(
        az.ess(samples, method="mean"), rel=1e-12
    )
    assert ours["bulk_split_rank_normalized"]["value"] == pytest.approx(
        az.ess(samples, method="bulk"), rel=1e-12
    )
    assert 6_500 < ours["identity_original_chains"]["value"] < 8_500
    assert result["status"] == "ok"


def test_ar1_ess_degrades_matching_arviz_and_theory():
    samples = ar1(phi=0.8, n_chains=4, n_draws=2_000, seed=31)
    result = analyze_chains(samples, default_max_acf_lag=10)
    ess_identity = result["effective_sample_size"]["identity_original_chains"]
    theoretical_ess = samples.size * (1 - 0.8) / (1 + 0.8)
    assert ess_identity["value"] == pytest.approx(
        az.ess(samples, method="identity"), rel=1e-12
    )
    assert ess_identity["value"] == pytest.approx(theoretical_ess, rel=0.18)
    assert ess_identity["integrated_autocorrelation_time"] > 4.0
    assert ess_identity["cutoff_lag"] > 1


def test_offset_means_separate_within_and_between_variance_and_flag_rhat():
    rng = np.random.default_rng(41)
    samples = np.vstack(
        [rng.normal(-4, 0.5, size=(2, 1_000)), rng.normal(4, 0.5, size=(2, 1_000))]
    )
    result = analyze_chains(samples)
    variance = result["variance"]["original_batches"]
    assert variance["within_batch_variance_W"] == pytest.approx(0.25, abs=0.03)
    assert variance["between_batch_variance_B_scaled_by_n"] > 10_000
    assert result["convergence"]["rank_split_chains"]["value"] == pytest.approx(
        az.rhat(samples, method="rank"), abs=1e-12
    )
    assert result["convergence"]["rank_split_chains"]["value"] > 1.5
    codes = [item["code"] for item in result["findings"]]
    assert "RHAT_ABOVE_THRESHOLD" in codes
    assert "ESS_IDENTITY_LOW" in codes
    assert result["status"] == "warning"


def test_constant_chains_are_indeterminate_instead_of_reported_as_informative():
    samples = np.full((2, 100), 5.0)
    result = analyze_chains(samples)
    codes = [item["code"] for item in result["findings"]]
    assert codes.count("CONSTANT_CHAIN") == 2
    assert result["effective_sample_size"]["identity_original_chains"] == {
        "value": None,
        "status": "not_estimated_constant",
        "reason": "All analyzed values are identical, so autocorrelation and ESS are undefined.",
    }
    assert result["convergence"]["classic_original_chains"]["status"] == (
        "not_estimated_constant"
    )
    assert result["chains"][0]["acf"]["values"] is None
    assert result["status"] == "inconclusive"


def test_constant_offset_chains_have_infinite_rhat_category_not_normal_rhat():
    samples = np.vstack([np.full((1, 10), 1.0), np.full((1, 10), 5.0)])
    result = analyze_chains(samples)
    classic = result["convergence"]["classic_original_chains"]
    rank = result["convergence"]["rank_split_chains"]
    assert classic["extended_value"] == "plus_infinity"
    assert rank["extended_value"] == "plus_infinity"
    assert classic["status"] == "inconclusive_zero_within_variance"
    assert "RHAT_PLUS_INFINITY_CONSTANT_BATCHES" in [
        item["code"] for item in result["findings"]
    ]


def test_short_chains_suppress_geyer_split_estimates_with_specific_category():
    samples = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    result = analyze_chains(samples)
    codes = [item["code"] for item in result["findings"]]
    assert codes.count("SHORT_CHAIN") == 2
    assert (
        result["effective_sample_size"]["mean_split_chains"]["status"]
        == "not_estimated_short_chain"
    )
    assert (
        result["convergence"]["rank_split_chains"]["status"]
        == "not_estimated_short_chain"
    )
    assert result["convergence"]["classic_original_chains"]["status"] == "estimated"


def test_single_chain_reports_cross_chain_diagnostic_limit_but_still_uses_acf():
    samples = ar1(phi=0.6, n_chains=1, n_draws=1_000, seed=51)
    result = analyze_chains(samples)
    codes = [item["code"] for item in result["findings"]]
    assert "SINGLE_CHAIN_CROSS_CHAIN_UNAVAILABLE" in codes
    assert result["convergence"]["rank_split_chains"]["status"] == (
        "not_estimated_single_chain"
    )
    assert result["convergence"]["within_chain_rank_split_for_single_chain"]["status"] == (
        "estimated"
    )
    assert result["status"] == "warning"


def test_unequal_chain_lengths_are_contract_failures():
    with pytest.raises(DiagnosticInputError) as exc:
        analyze_chains([[1.0, 2.0], [1.0]])
    assert exc.value.code == "EQUAL_LENGTH_REQUIRED"


def test_non_finite_input_has_specific_failure_category():
    with pytest.raises(DiagnosticInputError) as exc:
        analyze_chains([[1.0, np.nan]])
    assert exc.value.code == "NON_FINITE_SAMPLE"
