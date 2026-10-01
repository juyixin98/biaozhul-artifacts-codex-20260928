from __future__ import annotations

import math

import numpy as np
import pytest

from survey_imputation.errors import (
    AllOutcomeMissingError,
    InsufficientObservedRowsError,
    InvalidParameterError,
)
from survey_imputation.kernel import (
    build_design,
    complete_case,
    complete_data,
    mean_imputation,
    multiple_imputation,
)


def _minimal_arrays(minimal_fixture):
    rows = minimal_fixture["records"]
    y = np.array([np.nan if row["rating"] is None else row["rating"] for row in rows], float)
    x = np.array(
        [[row["age_z"], row["member"], row["used_support"], row["contact_frequency"]] for row in rows],
        float,
    )
    return build_design([row["record_id"] for row in rows], y, x, np.isnan(y))


def test_mean_imputation_matches_hand_calculated_fixture(minimal_fixture):
    design = _minimal_arrays(minimal_fixture)

    observed = complete_case(design)
    filled = mean_imputation(design)

    assert observed.estimate == pytest.approx(5.285714285714286)
    assert observed.variance == pytest.approx(13.904761904761905 / 7)
    assert observed.n_records == 10
    assert observed.marked_imputed == 0

    assert filled.estimate == pytest.approx(5.285714285714286)
    assert filled.variance == pytest.approx(0.926984126984127)
    assert filled.standard_error < observed.standard_error
    assert filled.marked_imputed == 3


def test_mean_imputation_keeps_real_zero_and_rejects_all_missing():
    ids = ["zero", "missing"]
    y = np.array([0.0, np.nan], dtype=float)
    x = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    design = build_design(ids, y, x, np.isnan(y))

    result = mean_imputation(design)
    assert result.estimate == pytest.approx(0.0)
    assert result.completed_outcomes[0] == 0.0

    all_missing = build_design(
        ["a", "b"], np.array([np.nan, np.nan]), np.zeros((2, 4)), np.array([True, True])
    )
    with pytest.raises(AllOutcomeMissingError) as exc_info:
        mean_imputation(all_missing)
    assert exc_info.value.category == "all_outcome_missing"


def test_multiple_imputation_independently_reimplements_rubin_formula(minimal_fixture):
    design = _minimal_arrays(minimal_fixture)
    result = multiple_imputation(design, m=4, seed=1729)

    # The following is an explicit independent implementation of the same
    # seeded posterior draw and Rubin pooling steps used for numerical checking.
    observed = ~design.mask
    x_with_intercept = design.x
    x_obs, y_obs = x_with_intercept[observed], design.y[observed]
    x_miss = x_with_intercept[design.mask]
    beta, *_ = np.linalg.lstsq(x_obs, y_obs, rcond=None)
    xtx_inv = np.linalg.inv(x_obs.T @ x_obs)
    residual_df = x_obs.shape[0] - x_obs.shape[1]
    sigma2_hat = np.sum((y_obs - x_obs @ beta) ** 2) / residual_df

    q_values = []
    u_values = []
    for stream_seed in np.random.SeedSequence(1729).spawn(4):
        rng = np.random.default_rng(stream_seed)
        sigma2 = residual_df * sigma2_hat / rng.chisquare(residual_df)
        beta_draw = rng.multivariate_normal(beta, sigma2 * xtx_inv)
        draws = np.clip(
            x_miss @ beta_draw + rng.normal(0, math.sqrt(sigma2), size=x_miss.shape[0]),
            0,
            10,
        )
        completed = design.y.copy()
        completed[design.mask] = draws
        q_values.append(completed.mean())
        u_values.append(completed.var(ddof=1) / completed.size)

    qbar = np.mean(q_values)
    ubar = np.mean(u_values)
    between = np.var(q_values, ddof=1)
    total = ubar + (1 + 1 / 4) * between
    lambda_m = ((1 + 1 / 4) * between) / total
    df_old = 3 / lambda_m**2
    df_complete = 10 - 1
    df_observed = df_complete * (df_complete + 3) / (df_complete + 1) * (1 - lambda_m)
    expected_df = df_old * df_observed / (df_old + df_observed)

    assert result.estimate == pytest.approx(qbar)
    assert result.variance == pytest.approx(total)
    assert result.degrees_of_freedom == pytest.approx(expected_df)
    assert result.standard_error == pytest.approx(math.sqrt(total))
    assert result.imputation_count == 4
    assert result.imputed_outcomes.shape == (4, 3)


def test_seed_replay_is_exact_and_stream_draws_differ(minimal_fixture):
    design = _minimal_arrays(minimal_fixture)
    first = multiple_imputation(design, m=5, seed=1729)
    second = multiple_imputation(design, m=5, seed=1729)
    other = multiple_imputation(design, m=5, seed=1730)

    np.testing.assert_allclose(first.imputed_outcomes, second.imputed_outcomes)
    assert not np.allclose(first.imputed_outcomes, other.imputed_outcomes)
    first_dataset_values = first.imputed_outcomes[:, 0]
    assert np.ptp(first_dataset_values) > 0.5


def test_complete_data_rejects_missing_but_accepts_known_complete_dataset():
    complete = build_design(
        ["a", "b"],
        np.array([0.0, 8.0]),
        np.array([[0, 0, 0, 0], [0, 1, 1, 2]]),
        np.array([False, False]),
    )
    assert complete_data(complete).estimate == 4.0

    missing = build_design(
        ["a", "b"],
        np.array([np.nan, 8.0]),
        np.array([[0, 0, 1, 0], [0, 1, 1, 2]]),
        np.array([True, False]),
    )
    with pytest.raises(InvalidParameterError) as exc_info:
        complete_data(missing)
    assert exc_info.value.category == "invalid_parameter"


def test_regression_requires_enough_allowed_observed_rows():
    y = np.array([1.0, 2.0, np.nan, np.nan])
    x = np.zeros((4, 4))
    x[:, 0] = 1
    design = build_design(["a", "b", "c", "d"], y, x, np.isnan(y))

    with pytest.raises(InsufficientObservedRowsError) as exc_info:
        multiple_imputation(design, m=3, seed=1)
    assert exc_info.value.category == "insufficient_observed_rows"
