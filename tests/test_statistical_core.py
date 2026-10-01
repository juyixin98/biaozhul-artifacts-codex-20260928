from __future__ import annotations

import numpy as np
import pytest

from survey_backend.errors import AllMissingFieldError, IneligibleObservationError
from survey_backend.statistical.analysis import run_comparison
from survey_backend.statistical.data import prepare_records
from survey_backend.statistical.imputation import mean_impute_predictors
from survey_backend.statistical.multiple import multiple_impute
from survey_backend.statistical.regression import fit_ols
from survey_backend.synthesis.generator import TrueParameters, generate_survey


def test_synthetic_generator_distinguishes_missing_from_true_zero_and_scope():
    survey = generate_survey(n=240, seed=11, mechanism="mar", missing_probability=0.15)
    allowed = survey.eligible

    assert survey.missing_mask.dtype == bool
    assert np.any(survey.missing_mask[allowed, 0])
    assert np.any((survey.q1 == 0.0) & survey.observed_q1 & allowed)
    assert not np.any(survey.missing_mask[~allowed])
    assert np.all(survey.q1[~allowed] == 30.0)

    records = survey.to_records()
    missing_records = [record for record in records if record.q1_screen_hours is None]
    zero_records = [record for record in records if record.q1_screen_hours == 0.0]
    assert missing_records
    assert zero_records
    assert all(record.eligible for record in missing_records)


def test_seed_replay_is_exact_and_mechanisms_produce_different_masks():
    first = generate_survey(n=320, seed=991, mechanism="mar")
    second = generate_survey(n=320, seed=991, mechanism="mar")
    other = generate_survey(n=320, seed=991, mechanism="mnar")

    np.testing.assert_array_equal(first.q1, second.q1)
    np.testing.assert_array_equal(first.missing_mask, second.missing_mask)
    assert not np.array_equal(first.missing_mask, other.missing_mask)


def test_known_parameter_full_data_baseline_recovers_coefficients_without_ineligible_rows():
    survey = generate_survey(
        n=1600,
        seed=808,
        mechanism="complete",
        ineligible_probability=0.15,
    )
    allowed_rows = np.flatnonzero(survey.eligible)
    x = np.column_stack(
        [
            survey.q1[allowed_rows],
            survey.q2[allowed_rows],
            np.ones(allowed_rows.size),
        ]
    )
    beta, _, _, _ = np.linalg.lstsq(x, survey.y[allowed_rows], rcond=None)

    np.testing.assert_allclose(
        beta,
        [TrueParameters.beta_q1, TrueParameters.beta_q2, TrueParameters.intercept],
        atol=0.12,
    )
    all_rows = np.column_stack([survey.q1, survey.q2, np.ones(survey.q1.size)])
    leaked_beta, _, _, _ = np.linalg.lstsq(all_rows, survey.y, rcond=None)
    assert np.max(np.abs(leaked_beta - beta)) > 0.5


def test_all_missing_allowed_field_is_rejected_with_specific_error():
    survey = generate_survey(n=80, seed=12, make_q1_all_missing=True)
    data = prepare_records(survey.to_records())

    with pytest.raises(AllMissingFieldError, match="q1_screen_hours"):
        run_comparison(
            data,
            analysis_id="a",
            version=1,
            dataset_id="d",
            request_id="r",
            request_hash="h",
            confidence=0.95,
            seed=1,
            imputations=2,
            burn_in=1,
        )


def test_no_eligible_records_has_distinct_failure_class():
    survey = generate_survey(n=40, seed=12, ineligible_probability=0.0)
    records = [record.model_copy(update={"eligible": False}) for record in survey.to_records()]

    with pytest.raises(IneligibleObservationError):
        prepare_records(records)


def test_mean_imputation_reference_is_independent_numpy_calculation():
    survey = generate_survey(
        n=40,
        seed=44,
        mechanism="mar",
        missing_probability=0.05,
        ineligible_probability=0.0,
    )
    data = prepare_records(survey.to_records())
    data.missing[0, 0] = True
    data.x[0, 0] = np.nan
    data.missing[0, 1] = False
    data.missing[0, 2] = False

    completed = mean_impute_predictors(data)
    expected_q1 = float(np.mean(survey.q1[1:]))
    assert completed.x[0, 0] == pytest.approx(expected_q1)
    expected_x = np.column_stack([completed.x[:, 0], completed.x[:, 1], np.ones(40)])
    expected_beta, _, _, _ = np.linalg.lstsq(expected_x, completed.y, rcond=None)
    ols = fit_ols(completed.x, completed.y)
    np.testing.assert_allclose(ols.beta, expected_beta, rtol=1e-10)


def test_multiple_imputation_replays_and_uses_independent_streams_with_rubin_pooling():
    survey = generate_survey(n=160, seed=77, mechanism="mar", missing_probability=0.18)
    data = prepare_records(survey.to_records())
    first = multiple_impute(data, m=4, burn_in=3, seed=303)
    second = multiple_impute(data, m=4, burn_in=3, seed=303)
    other = multiple_impute(data, m=4, burn_in=3, seed=304)

    np.testing.assert_allclose(first.pooled_beta, second.pooled_beta)
    np.testing.assert_allclose(first.total_variance, second.total_variance)
    assert not np.allclose(first.pooled_beta, other.pooled_beta, atol=1e-8)
    assert first.stream_tags[0] != first.stream_tags[1]
    assert not np.allclose(first.betas[0], first.betas[1])

    # Independent direct Rubin formula check for the first coefficient.
    q = first.betas[:, 0]
    u = np.diagonal(first.covariances)[:, 0]
    m = 4
    ubar = float(np.mean(u))
    b = float(np.sum((q - np.mean(q)) ** 2) / (m - 1))
    expected_total = ubar + (1 + 1 / m) * b
    assert first.total_variance[0] == pytest.approx(expected_total)


def test_methods_differ_under_mar_and_mnar():
    response = run_comparison(
        prepare_records(
            generate_survey(
                n=500,
                seed=515,
                mechanism="mnar",
                missing_probability=0.25,
            ).to_records()
        ),
        analysis_id="a",
        version=1,
        dataset_id="d",
        request_id="r",
        request_hash="h",
        confidence=0.95,
        seed=515,
        imputations=3,
        burn_in=2,
    )
    vectors = {
        result.method: np.asarray([c.estimate for c in result.coefficients])
        for result in response.results
    }
    assert not np.allclose(vectors["complete_case"], vectors["mean_imputation"], atol=1e-8)
    assert not np.allclose(vectors["mean_imputation"], vectors["multiple_imputation"], atol=1e-8)
