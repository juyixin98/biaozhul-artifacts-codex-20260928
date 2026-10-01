from __future__ import annotations

import numpy as np

from survey_imputation.kernel import build_design, run_method
from survey_imputation.synthetic import TRUE_PARAMETERS, generate_survey


def test_synthetic_dataset_replays_seed_and_separates_zeros_from_missing():
    first = generate_survey(n=200, mechanism="mar", missing_probability=0.2, seed=1729)
    second = generate_survey(n=200, mechanism="mar", missing_probability=0.2, seed=1729)

    assert first.n_missing == 42
    assert second.n_missing == 42
    assert [r.model_dump() for r in first.records] == [r.model_dump() for r in second.records]
    assert first.true_mean_rating == 5.368195000000001

    missing_records = [r for r in first.records if r.rating is None]
    zero_records = [r for r in first.records if r.rating == 0]
    assert len(missing_records) == 42
    assert len(zero_records) == 48
    assert all(r.used_support == 1 for r in missing_records)
    assert all(r.used_support == 0 for r in zero_records)


def test_known_parameters_are_recovered_on_allowed_observed_user_rows():
    generated = generate_survey(n=1000, mechanism="mcar", missing_probability=0.2, seed=20260930)
    eligible = [
        record
        for record in generated.records
        if record.used_support == 1 and record.rating is not None
    ]
    x = np.column_stack(
        [
            np.ones(len(eligible)),
            [r.age_z for r in eligible],
            [r.member for r in eligible],
            [r.contact_frequency for r in eligible],
        ]
    )
    y = np.asarray([r.rating for r in eligible], dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)

    # Among users the model is 7.25 - 0.65 + 0.20 age + 0.45 member + 0.08 contact + noise.
    assert 6.35 <= beta[0] <= 6.85
    assert abs(beta[1] - 0.20) <= 0.08
    assert abs(beta[2] - 0.45) <= 0.12
    assert abs(beta[3] - 0.08) <= 0.04
    residual_sd = np.sqrt(np.sum((y - x @ beta) ** 2) / (x.shape[0] - x.shape[1]))
    assert abs(residual_sd - TRUE_PARAMETERS["noise_sd"]) <= 0.08


def test_method_comparison_shows_variance_and_mnar_bias_difference():
    generated = generate_survey(n=1000, mechanism="mnar", missing_probability=0.2, seed=20260930)
    y = np.asarray([np.nan if r.rating is None else r.rating for r in generated.records], float)
    x = np.asarray(
        [[r.age_z, r.member, r.used_support, r.contact_frequency] for r in generated.records],
        float,
    )
    design = build_design([r.record_id for r in generated.records], y, x, np.isnan(y))

    complete_case = run_method(design, "complete_case")
    mean = run_method(design, "mean_imputation")
    mi = run_method(design, "multiple_imputation", m=30, seed=20261010)

    assert abs(complete_case.estimate - mean.estimate) < 1e-12
    assert mean.standard_error < complete_case.standard_error
    assert abs(mi.estimate - generated.true_mean_rating) < abs(
        mean.estimate - generated.true_mean_rating
    )
    assert abs(mi.estimate - generated.true_mean_rating) > 0.05
    assert any(item.code == "mar_assumption_status" for item in generated.diagnostics)
