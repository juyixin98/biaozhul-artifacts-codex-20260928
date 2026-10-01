from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from survey_backend.errors import SingularDesignError
from survey_backend.statistical.data import design_matrix


@dataclass(frozen=True)
class OLSResult:
    beta: np.ndarray
    covariance: np.ndarray
    residual_variance: float
    n_used: int
    rank: int


def fit_ols(x: np.ndarray, y: np.ndarray) -> OLSResult:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y dimensions do not match")
    n_used, p = x.shape
    if n_used <= p:
        raise SingularDesignError(f"OLS requires more than {p} complete observations")

    x_design = design_matrix(x)
    try:
        beta, residuals, rank, _ = np.linalg.lstsq(x_design, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise SingularDesignError("design matrix could not be solved") from exc

    expected_rank = x_design.shape[1]
    if rank != expected_rank:
        raise SingularDesignError("design matrix is rank deficient")

    fitted = x_design @ beta
    residual = y - fitted
    degrees_of_freedom = n_used - expected_rank
    residual_variance = float(residual @ residual / degrees_of_freedom)
    gram = x_design.T @ x_design
    try:
        covariance = residual_variance * np.linalg.inv(gram)
    except np.linalg.LinAlgError as exc:
        raise SingularDesignError("design covariance could not be inverted") from exc
    if not np.all(np.isfinite(covariance)):
        raise SingularDesignError("design covariance contains non-finite values")
    return OLSResult(
        beta=beta,
        covariance=covariance,
        residual_variance=residual_variance,
        n_used=n_used,
        rank=rank,
    )


def coefficient_summaries(
    beta: np.ndarray,
    covariance: np.ndarray,
    confidence: float,
    degrees_of_freedom: float | np.ndarray | None,
    names: tuple[str, str, str],
) -> list[dict[str, float | str | None]]:
    alpha = 1.0 - confidence
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if degrees_of_freedom is None:
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        lower = beta - z * standard_errors
        upper = beta + z * standard_errors
        p_values = 2.0 * stats.norm.sf(np.abs(beta / standard_errors))
    else:
        dfs = np.broadcast_to(np.asarray(degrees_of_freedom, dtype=float), beta.shape)
        t_critical = stats.t.ppf(1.0 - alpha / 2.0, dfs)
        lower = beta - t_critical * standard_errors
        upper = beta + t_critical * standard_errors
        p_values = np.asarray(
            [2.0 * stats.t.sf(abs(z_score), dfs[i]) for i, z_score in enumerate(beta / standard_errors)]
        )
    return [
        {
            "name": name,
            "estimate": float(beta[i]),
            "variance": float(covariance[i, i]),
            "standard_error": float(standard_errors[i]),
            "p_value": float(p_values[i]),
            "ci_lower": float(lower[i]),
            "ci_upper": float(upper[i]),
        }
        for i, name in enumerate(names)
    ]
