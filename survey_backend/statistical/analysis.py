from __future__ import annotations

import numpy as np

from survey_backend.models import (
    AnalysisResponse,
    CoefficientResult,
    Diagnostic,
    ImputationRun,
    MethodResult,
    PoolingDetail,
)
from survey_backend.statistical.data import COEFFICIENT_NAMES, AnalysisData
from survey_backend.statistical.diagnostics import make_diagnostic, request_diagnostics
from survey_backend.statistical.imputation import mean_impute_predictors
from survey_backend.statistical.multiple import multiple_impute
from survey_backend.statistical.regression import coefficient_summaries, fit_ols
from survey_backend.statistical.validation import complete_case_rows


ASSUMPTIONS = [
    "Allowed observations are identified by the eligible mask before any model fitting.",
    "Null means missing and 0.0 means a genuine observed zero.",
    "Complete-case regression is unbiased when all model variables are MCAR.",
    "Mean imputation is a single-valued benchmark; it attenuates variance and can be biased under MAR.",
    "Multiple imputation assumes ignorable MAR, a conditional-linear Gaussian model, and independent random streams.",
    "No missingness mechanism test can identify MNAR from the observed data; MNAR bias is not guaranteed to vanish.",
    "Imputed values are retained as synthetic draws and are never counted as real observations.",
]


def _to_coefficients(
    beta: np.ndarray,
    covariance: np.ndarray,
    confidence: float,
    degrees_of_freedom: float | np.ndarray | None,
) -> list[CoefficientResult]:
    return [
        CoefficientResult(**row)
        for row in coefficient_summaries(
            beta, covariance, confidence, degrees_of_freedom, COEFFICIENT_NAMES
        )
    ]


def _base_kwargs(data: AnalysisData) -> dict[str, object]:
    allowed_n = int(np.sum(data.eligible))
    return {
        "used_observations": 0,
        "dropped_ineligible": data.n - allowed_n,
        "missing_counts": data.missing_counts_allowed(),
        "imputed_counts": {"q1_screen_hours": 0, "q2_sleep_hours": 0},
    }


def run_complete_case(
    data: AnalysisData, confidence: float, request_id: str
) -> MethodResult:
    rows = complete_case_rows(data)
    ols = fit_ols(data.x[rows, :], data.y[rows])
    kwargs = _base_kwargs(data)
    kwargs.update(
        method="complete_case",
        status="accepted",
        used_observations=int(rows.size),
        coefficients=_to_coefficients(
            ols.beta, ols.covariance, confidence, float(rows.size - 3)
        ),
        residual_variance=ols.residual_variance,
        diagnostics=[
            make_diagnostic(
                "accepted",
                "complete_case_accepted",
                "Accepted because all allowed rows used in the fit have q1, q2, and satisfaction observed.",
                {"complete_rows": int(rows.size), "assumption": "MCAR for unbiasedness"},
                request_id=request_id,
            )
        ],
    )
    return MethodResult(**kwargs)


def run_mean_imputation(
    data: AnalysisData, confidence: float, request_id: str
) -> MethodResult:
    completed = mean_impute_predictors(data)
    ols = fit_ols(completed.x, completed.y)
    kwargs = _base_kwargs(data)
    kwargs.update(
        method="mean_imputation",
        status="accepted",
        used_observations=completed.x.shape[0],
        imputed_counts={
            "q1_screen_hours": completed.imputed_counts[0],
            "q2_sleep_hours": completed.imputed_counts[1],
        },
        coefficients=_to_coefficients(
            ols.beta,
            ols.covariance,
            confidence,
            float(completed.x.shape[0] - 3),
        ),
        residual_variance=ols.residual_variance,
        diagnostics=[
            make_diagnostic(
                "accepted",
                "mean_imputation_accepted",
                "Accepted as a benchmark using allowed observed-outcome rows; missing predictors receive conditional allowed means.",
                {
                    "observed_outcome_rows": completed.x.shape[0],
                    "imputed_q1": completed.imputed_counts[0],
                    "imputed_q2": completed.imputed_counts[1],
                    "imputed_values_are_observations": False,
                },
                request_id=request_id,
            ),
            make_diagnostic(
                "undetermined",
                "mean_imputation_uncertainty",
                "The single fill has no between-imputation variance; standard errors can be too small.",
                {"variance_adjusted": False},
                request_id=request_id,
            ),
        ],
    )
    return MethodResult(**kwargs)


def run_multiple_imputation(
    data: AnalysisData,
    confidence: float,
    request_id: str,
    m: int,
    burn_in: int,
    seed: int,
) -> MethodResult:
    pooled = multiple_impute(data, m=m, burn_in=burn_in, seed=seed)
    runs: list[ImputationRun] = []
    residual_variances = []
    for index, completed in enumerate(pooled.completed_datasets):
        ols = fit_ols(completed.x, completed.y)
        residual_variances.append(ols.residual_variance)
        runs.append(
            ImputationRun(
                imputation_index=index + 1,
                stream_tag=pooled.stream_tags[index],
                imputed_q1=completed.imputed_counts[0],
                imputed_q2=completed.imputed_counts[1],
                coefficients=[CoefficientResult(**row) for row in coefficient_summaries(
                    ols.beta,
                    ols.covariance,
                    confidence,
                    float(completed.x.shape[0] - 3),
                    COEFFICIENT_NAMES,
                )],
                residual_variance=ols.residual_variance,
            )
        )

    dfs = [None if not np.isfinite(value) else float(value) for value in pooled.degrees_of_freedom]
    pooling = [
        PoolingDetail(
            parameter=COEFFICIENT_NAMES[i],
            within_variance=float(pooled.within_variance[i]),
            between_variance=float(pooled.between_variance[i]),
            total_variance=float(pooled.total_variance[i]),
            relative_increase_variance=float(pooled.relative_increase_variance[i]),
            fraction_missing_information=float(pooled.fraction_missing_information[i]),
            degrees_of_freedom=dfs[i],
        )
        for i in range(len(COEFFICIENT_NAMES))
    ]
    first = pooled.completed_datasets[0]
    kwargs = _base_kwargs(data)
    kwargs.update(
        method="multiple_imputation",
        status="accepted",
        used_observations=first.x.shape[0],
        imputed_counts={
            "q1_screen_hours": first.imputed_counts[0],
            "q2_sleep_hours": first.imputed_counts[1],
        },
        coefficients=_to_coefficients(
            pooled.pooled_beta,
            pooled.pooled_covariance,
            confidence,
            np.asarray([np.inf if value is None else value for value in dfs]),
        ),
        residual_variance=float(np.mean(residual_variances)),
        pooling=pooling,
        runs=runs,
        diagnostics=[
            make_diagnostic(
                "accepted",
                "multiple_imputation_accepted",
                "Accepted after MICE fills, per-dataset OLS estimates, and Rubin pooling; imputed draws remain synthetic.",
                {
                    "m": m,
                    "burn_in": burn_in,
                    "seed": seed,
                    "observed_outcome_rows": first.x.shape[0],
                    "imputed_q1": first.imputed_counts[0],
                    "imputed_q2": first.imputed_counts[1],
                    "imputed_values_are_observations": False,
                    "independent_streams": True,
                },
                request_id=request_id,
            )
        ],
    )
    return MethodResult(**kwargs)


def run_comparison(
    data: AnalysisData,
    *,
    analysis_id: str,
    version: int,
    dataset_id: str,
    request_id: str,
    request_hash: str,
    confidence: float,
    seed: int,
    imputations: int,
    burn_in: int,
    methods: list[str] | None = None,
) -> AnalysisResponse:
    selected = methods or [
        "complete_case",
        "mean_imputation",
        "multiple_imputation",
    ]
    allowed_n = int(np.sum(data.eligible))
    base_diagnostics = request_diagnostics(
        data,
        request_id=request_id,
        missing_counts=data.missing_counts_allowed(),
        zero_counts=data.zero_counts_allowed(),
        allowed_n=allowed_n,
    )
    results: list[MethodResult] = []
    for method in selected:
        if method == "complete_case":
            result = run_complete_case(data, confidence, request_id)
        elif method == "mean_imputation":
            result = run_mean_imputation(data, confidence, request_id)
        else:
            result = run_multiple_imputation(
                data, confidence, request_id, imputations, burn_in, seed
            )
        result.diagnostics = base_diagnostics + result.diagnostics
        results.append(result)

    return AnalysisResponse(
        analysis_id=analysis_id,
        version=version,
        dataset_id=dataset_id,
        request_id=request_id,
        request_hash=request_hash,
        confidence=confidence,
        seed=seed,
        imputations=imputations,
        burn_in=burn_in,
        assumptions=ASSUMPTIONS,
        results=results,
    )
