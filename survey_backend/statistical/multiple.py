from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from survey_backend.errors import SingularDesignError
from survey_backend.statistical.data import AnalysisData, design_matrix
from survey_backend.statistical.imputation import CompletedData
from survey_backend.statistical.validation import (
    observed_predictor_counts,
    observed_outcome_rows,
    reject_rows_with_all_predictors_missing,
)


@dataclass(frozen=True)
class MultipleImputationResult:
    completed_datasets: list[CompletedData]
    betas: np.ndarray
    covariances: np.ndarray
    pooled_beta: np.ndarray
    pooled_covariance: np.ndarray
    within_variance: np.ndarray
    between_variance: np.ndarray
    total_variance: np.ndarray
    degrees_of_freedom: np.ndarray
    relative_increase_variance: np.ndarray
    fraction_missing_information: np.ndarray
    stream_tags: list[str]


def _draw_bayesian_conditional(
    predictors: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    missing: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    x_obs = design_matrix(predictors[observed])
    y_obs = target[observed]
    n_obs, p = x_obs.shape
    if n_obs <= p:
        raise SingularDesignError("not enough observed rows for a Bayesian imputation regression")
    beta_hat, _, rank, _ = np.linalg.lstsq(x_obs, y_obs, rcond=None)
    if rank != p:
        raise SingularDesignError("Bayesian imputation design is rank deficient")
    residual = y_obs - x_obs @ beta_hat
    sse = float(residual @ residual)
    chi2 = rng.chisquare(max(n_obs - p, 1))
    sigma2 = sse / chi2
    info = x_obs.T @ x_obs
    try:
        beta = rng.multivariate_normal(beta_hat, sigma2 * np.linalg.inv(info))
    except np.linalg.LinAlgError as exc:
        raise SingularDesignError("Bayesian posterior covariance is singular") from exc

    completed = target.copy()
    if np.any(missing):
        x_missing = design_matrix(predictors[missing])
        completed[missing] = x_missing @ beta + rng.normal(
            scale=np.sqrt(sigma2), size=int(np.sum(missing))
        )
    return completed


def impute_once(
    data: AnalysisData,
    rows: np.ndarray,
    rng: np.random.Generator,
    burn_in: int,
    imputation_index: int,
) -> CompletedData:
    x = data.x[rows, :].copy()
    y = data.y[rows].copy()
    missing_x = data.missing[rows, :2].copy()
    for j in range(2):
        observed = ~missing_x[:, j]
        x[~observed, j] = float(rng.normal(np.mean(x[observed, j]), np.std(x[observed, j])))

    # MICE with two chained Bayesian linear regressions. The intercept is
    # implicit, and only allowed observed-outcome rows enter every fit.
    for _ in range(burn_in):
        x[:, 0] = _draw_bayesian_conditional(
            predictors=np.column_stack([x[:, 1], y]),
            target=x[:, 0],
            observed=~missing_x[:, 0],
            missing=missing_x[:, 0],
            rng=rng,
        )
        x[:, 1] = _draw_bayesian_conditional(
            predictors=np.column_stack([x[:, 0], y]),
            target=x[:, 1],
            observed=~missing_x[:, 1],
            missing=missing_x[:, 1],
            rng=rng,
        )

    return CompletedData(
        x=x,
        y=y,
        record_indices=rows,
        imputed_counts=(int(np.sum(missing_x[:, 0])), int(np.sum(missing_x[:, 1]))),
        is_imputed=missing_x,
        stream_tag=f"m{imputation_index + 1}",
    )


def pool_rubin(
    betas: np.ndarray,
    covariances: np.ndarray,
    stream_tags: list[str],
    completed_datasets: list[CompletedData],
) -> MultipleImputationResult:
    m = betas.shape[0]
    p = betas.shape[1]
    qbar = np.mean(betas, axis=0)
    ubar = np.mean(covariances, axis=0)
    deviations = betas - qbar
    between = (deviations.T @ deviations) / (m - 1)
    total = ubar + (1.0 + 1.0 / m) * between

    u_diag = np.maximum(np.diag(ubar), np.finfo(float).tiny)
    b_diag = np.maximum(np.diag(between), 0.0)
    r = (1.0 + 1.0 / m) * b_diag / u_diag
    df = np.full(p, np.inf)
    finite_b = b_diag > np.finfo(float).eps
    df[finite_b] = (m - 1.0) * (1.0 + u_diag[finite_b] / ((1.0 + 1.0 / m) * b_diag[finite_b])) ** 2
    fraction = r / (r + 1.0)
    finite_df = np.isfinite(df)
    fraction[finite_df] = (r[finite_df] + 2.0 / df[finite_df]) / (r[finite_df] + 1.0)
    fraction = np.clip(fraction, 0.0, 1.0)
    return MultipleImputationResult(
        completed_datasets=completed_datasets,
        betas=betas,
        covariances=covariances,
        pooled_beta=qbar,
        pooled_covariance=total,
        within_variance=u_diag,
        between_variance=b_diag,
        total_variance=np.diag(total),
        degrees_of_freedom=np.where(np.isfinite(df), df, np.nan),
        relative_increase_variance=r,
        fraction_missing_information=fraction,
        stream_tags=stream_tags,
    )


def multiple_impute(
    data: AnalysisData,
    m: int,
    burn_in: int,
    seed: int,
) -> MultipleImputationResult:
    rows = observed_outcome_rows(data)
    observed_predictor_counts(data, rows)
    reject_rows_with_all_predictors_missing(data, rows)
    child_seeds = np.random.SeedSequence(seed).spawn(m)
    completed_datasets: list[CompletedData] = []
    stream_tags: list[str] = []
    betas = []
    covariances = []
    for i, child_seed in enumerate(child_seeds):
        rng = np.random.default_rng(child_seed)
        completed = impute_once(data, rows, rng, burn_in, i)
        completed_datasets.append(completed)
        stream_tags.append(f"m{i + 1}:{child_seed.generate_state(1)[0]}")

        from survey_backend.statistical.regression import fit_ols

        ols = fit_ols(completed.x, completed.y)
        betas.append(ols.beta)
        covariances.append(ols.covariance)
    return pool_rubin(
        betas=np.asarray(betas),
        covariances=np.asarray(covariances),
        stream_tags=stream_tags,
        completed_datasets=completed_datasets,
    )
