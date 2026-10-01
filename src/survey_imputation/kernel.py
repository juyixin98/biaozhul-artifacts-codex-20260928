"""Statistical kernel for mean and multiple imputation.

The multiple imputation model is Bayesian normal linear regression. Training
uses only rows with observed ratings. Every imputation gets an independent
NumPy random stream, and point estimates are pooled with Rubin's rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import stats

from .contracts import DesignMatrices
from .errors import (
    AllOutcomeMissingError,
    InsufficientObservedRowsError,
    InvalidParameterError,
)
from .masking import missing_mask
from .versioning import CORE_VERSION, MODEL_VERSION


FEATURE_NAMES = ("intercept", "age_z", "member", "used_support", "contact_frequency")


@dataclass(frozen=True)
class MethodEstimate:
    method: str
    model_version: str
    n_records: int
    n_missing: int
    estimate: float
    variance: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    degrees_of_freedom: float | None
    imputation_count: int | None
    marked_imputed: int
    note: str
    completed_outcomes: np.ndarray | None = None
    imputed_outcomes: np.ndarray | None = None


def build_design(
    record_ids: list[str],
    y: np.ndarray,
    covariates: np.ndarray,
    provided_mask: np.ndarray | None = None,
) -> DesignMatrices:
    y = np.asarray(y, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim != 2 or covariates.shape[0] != y.shape[0]:
        raise InvalidParameterError("covariate matrix must have one row per record")
    x = np.column_stack([np.ones(y.shape[0]), covariates])
    if x.shape[1] != len(FEATURE_NAMES):
        raise InvalidParameterError("covariate matrix must contain four columns")
    mask = missing_mask(y, provided_mask)
    return DesignMatrices(list(record_ids), y, x, mask, FEATURE_NAMES)


def _t_interval(mean: float, variance: float, df: float) -> tuple[float, float]:
    if variance < 0.0 or math.isnan(variance):
        raise InvalidParameterError("estimated variance must be finite and non-negative")
    se = math.sqrt(max(variance, 0.0))
    if not math.isfinite(se):
        raise InvalidParameterError("standard error must be finite")
    critical = float(stats.t.ppf(0.975, df)) if df != math.inf else 1.959963984540054
    return mean - critical * se, mean + critical * se


def _mean_estimate(
    method: str,
    model_version: str,
    values: np.ndarray,
    n_total: int,
    n_missing: int,
    marked_imputed: int,
    note: str,
) -> MethodEstimate:
    n = values.size
    mean = float(np.mean(values))
    if n <= 1:
        raise InsufficientObservedRowsError("at least two values are required to estimate variance")
    sample_variance = float(np.var(values, ddof=1))
    variance = sample_variance / n
    low, high = _t_interval(mean, variance, n - 1)
    return MethodEstimate(
        method=method,
        model_version=model_version,
        n_records=n_total,
        n_missing=n_missing,
        estimate=mean,
        variance=variance,
        standard_error=math.sqrt(variance),
        ci95_low=low,
        ci95_high=high,
        degrees_of_freedom=float(n - 1),
        imputation_count=None,
        marked_imputed=marked_imputed,
        note=note,
        completed_outcomes=values,
    )


def complete_data(design: DesignMatrices) -> MethodEstimate:
    if np.any(design.mask):
        raise InvalidParameterError("complete_data requires a dataset with no missing outcomes")
    return _mean_estimate(
        "complete_data",
        "oracle-complete-data-1.0.0",
        design.y,
        design.y.size,
        0,
        0,
        "Observed complete-data baseline; no imputation is performed.",
    )


def complete_case(design: DesignMatrices) -> MethodEstimate:
    values = design.y[~design.mask]
    if values.size == 0:
        raise AllOutcomeMissingError("all ratings are missing, so complete-case analysis is impossible")
    return _mean_estimate(
        "complete_case",
        "complete-case-1.0.0",
        values,
        design.y.size,
        int(design.mask.sum()),
        0,
        "Rows with missing ratings are excluded; no values are imputed.",
    )


def mean_imputation(design: DesignMatrices) -> MethodEstimate:
    observed = design.y[~design.mask]
    if observed.size == 0:
        raise AllOutcomeMissingError("all ratings are missing, so the observed mean cannot be estimated")
    fill = float(np.mean(observed))
    completed = design.y.copy()
    completed[design.mask] = fill
    return _mean_estimate(
        "mean_imputation",
        MODEL_VERSION["mean_imputation"],
        completed,
        design.y.size,
        int(design.mask.sum()),
        int(design.mask.sum()),
        "Missing cells are filled once with the observed arithmetic mean. Filled cells remain "
        "marked and are not treated as independent observations; the naive completed-data "
        "variance is reported for comparison and typically understates uncertainty.",
    )


def _fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if x.shape[0] < x.shape[1]:
        raise InsufficientObservedRowsError(
            f"regression needs at least {x.shape[1]} observed rows, received {x.shape[0]}"
        )
    beta, residuals, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise InvalidParameterError("predictor matrix is rank-deficient in observed rows")
    rss = float(np.sum((y - x @ beta) ** 2))
    df = x.shape[0] - x.shape[1]
    if df <= 0:
        raise InsufficientObservedRowsError("not enough observed rows to estimate residual variance")
    xtx_inv = np.linalg.inv(x.T @ x)
    return beta, xtx_inv, rss / df


def _barnard_rubin_df(
    m: int,
    u_bar: float,
    between_variance: float,
    n_complete: int,
    k: int = 1,
) -> float:
    increment = (1.0 + 1.0 / m) * between_variance
    if increment <= 0.0:
        return float("inf")
    fraction_missing_info = increment / (u_bar + increment)
    df_old = (m - 1) / max(fraction_missing_info**2, np.finfo(float).tiny)
    df_complete = max(n_complete - k, 1)
    df_obs = df_complete * (df_complete + 3) / (df_complete + 1)
    df_obs *= 1.0 - fraction_missing_info
    df_obs = max(df_obs, np.finfo(float).tiny)
    return (df_old * df_obs) / (df_old + df_obs)


def multiple_imputation(design: DesignMatrices, m: int = 20, seed: int = 20260930) -> MethodEstimate:
    if m < 2:
        raise InvalidParameterError("multiple imputation requires m >= 2")

    observed_index = ~design.mask
    x_obs, y_obs = design.x[observed_index], design.y[observed_index]
    x_miss = design.x[design.mask]
    n_missing = int(design.mask.sum())
    if y_obs.size == 0:
        raise AllOutcomeMissingError("all ratings are missing, so imputation model parameters cannot be trained")

    beta_hat, xtx_inv, sigma2_hat = _fit_linear_regression(x_obs, y_obs)
    residual_df = x_obs.shape[0] - x_obs.shape[1]
    root_seed = np.random.SeedSequence(seed)
    child_seeds = root_seed.spawn(m)

    completed_datasets = []
    q_values = []
    u_values = []
    for stream_seed in child_seeds:
        rng = np.random.default_rng(stream_seed)
        sigma2 = residual_df * sigma2_hat / rng.chisquare(residual_df)
        beta = rng.multivariate_normal(beta_hat, sigma2 * xtx_inv)
        if n_missing:
            draws = x_miss @ beta + rng.normal(0.0, math.sqrt(sigma2), size=n_missing)
            draws = np.clip(draws, 0.0, 10.0)
            completed = design.y.copy()
            completed[design.mask] = draws
        else:
            completed = design.y.copy()
        completed_datasets.append(completed)
        q_values.append(float(np.mean(completed)))
        u_values.append(float(np.var(completed, ddof=1) / completed.size))

    q_bar = float(np.mean(q_values))
    u_bar = float(np.mean(u_values))
    between = float(np.var(q_values, ddof=1))
    total_variance = u_bar + (1.0 + 1.0 / m) * between
    df = _barnard_rubin_df(m, u_bar, between, design.y.size)
    low, high = _t_interval(q_bar, total_variance, df)
    return MethodEstimate(
        method="multiple_imputation",
        model_version=MODEL_VERSION["multiple_imputation"],
        n_records=design.y.size,
        n_missing=n_missing,
        estimate=q_bar,
        variance=total_variance,
        standard_error=math.sqrt(total_variance),
        ci95_low=low,
        ci95_high=high,
        degrees_of_freedom=df,
        imputation_count=m,
        marked_imputed=n_missing * m,
        note=(
            "Independent Bayesian regression draws are pooled using Qbar, Ubar, "
            "B, T=Ubar+(1+1/m)B, and a Barnard-Rubin t interval. Imputations are "
            "uncertain predictions, not observed ratings."
        ),
        completed_outcomes=None,
        imputed_outcomes=np.asarray([data[design.mask] for data in completed_datasets]),
    )


def run_method(design: DesignMatrices, method: str, m: int = 20, seed: int = 20260930) -> MethodEstimate:
    if method == "complete_data":
        return complete_data(design)
    if method == "complete_case":
        return complete_case(design)
    if method == "mean_imputation":
        return mean_imputation(design)
    if method == "multiple_imputation":
        return multiple_imputation(design, m=m, seed=seed)
    raise InvalidParameterError(f"unknown method: {method}")


__all__ = ["CORE_VERSION", "FEATURE_NAMES", "DesignMatrices", "MethodEstimate", "build_design", "run_method"]
