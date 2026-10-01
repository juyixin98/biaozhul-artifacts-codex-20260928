"""Covariance estimators with explicit algorithms and audit metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy import stats

from app.covariance.sample import MethodName, PreparedSample, robust_location_scale
from app.covariance.validation import MatrixDiagnostics, validate_covariance
from app.errors import EstimationError


@dataclass(frozen=True)
class SampleWeight:
    index: int
    weight: float
    robust_distance: float | None
    threshold: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "weight": self.weight,
            "robust_distance": self.robust_distance,
            "threshold": self.threshold,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EstimationResult:
    method: str
    covariance: np.ndarray
    location: list[float]
    weights: list[SampleWeight]
    diagnostics: MatrixDiagnostics
    parameters: dict[str, Any]
    assumptions: list[str]
    steps: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "location": self.location,
            "covariance": self.covariance.tolist(),
            "weights": [weight.to_dict() for weight in self.weights],
            "matrix_diagnostics": self.diagnostics.to_dict(),
            "parameters": self.parameters,
            "assumptions": self.assumptions,
            "algorithm_steps": self.steps,
            "warnings": self.warnings,
        }


def _steps(start: float, names: list[tuple[str, str]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for name, detail in names:
        steps.append(
            {
                "name": name,
                "detail": detail,
                "elapsed_ms": round((perf_counter() - start) * 1000.0, 3),
            }
        )
    return steps


def _unit_weights(n_samples: int, reason: str) -> list[SampleWeight]:
    return [
        SampleWeight(index=i, weight=1.0, robust_distance=None, threshold=None, reason=reason)
        for i in range(n_samples)
    ]


def empirical_covariance(sample: PreparedSample) -> EstimationResult:
    """Maximum-likelihood sample covariance with divisor n."""

    started = perf_counter()
    x = sample.values
    location = x.mean(axis=0)
    centered = x - location
    covariance = centered.T @ centered / x.shape[0]
    diagnostics = validate_covariance(covariance, method=MethodName.EMPIRICAL)
    assumptions = [
        "Rows are independent and identically distributed with finite second moments.",
        "The arithmetic mean is the assumed location; this estimator is not resistant to outliers.",
        "A rank below min(n_samples - 1, n_features) signals exact or numerical linear dependence.",
    ]
    warnings = ["No shrinkage is applied; the covariance can be singular when n_samples <= n_features."]
    if x.shape[0] <= x.shape[1]:
        warnings.append("n_samples <= n_features: ordinary sample covariance is generally singular.")
    return EstimationResult(
        method=MethodName.EMPIRICAL,
        covariance=covariance,
        location=location.tolist(),
        weights=_unit_weights(x.shape[0], "Empirical covariance assigns equal weight to every row."),
        diagnostics=diagnostics,
        parameters={"divisor": "n_samples", "center": "arithmetic_mean", "column_scaling": "none"},
        assumptions=assumptions,
        steps=_steps(
            started,
            [
                ("center", "Subtract each column's arithmetic mean."),
                ("scatter", "Compute X'X divided by n_samples."),
                ("validate", "Independently check symmetry, PSD, SVD rank, and invertibility."),
            ],
        ),
        warnings=warnings,
    )


def ledoit_wolf_covariance(sample: PreparedSample) -> EstimationResult:
    """Ledoit-Wolf shrinkage toward an equal-variance scalar matrix.

    The intensity uses the finite-sample formula from O. Ledoit and M. Wolf,
    "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices"
    (2004).  The divisor is n, matching that estimator's MLE covariance.
    """

    started = perf_counter()
    x = sample.values
    n, p = x.shape
    location = x.mean(axis=0)
    centered = x - location
    sample_cov = centered.T @ centered / n
    target_variance = float(np.trace(sample_cov) / p)
    target = target_variance * np.eye(p)

    target_deviation = sample_cov.copy()
    np.fill_diagonal(target_deviation, np.diag(target_deviation) - target_variance)
    delta = float(np.sum(target_deviation**2) / p)

    squared = centered**2
    beta_matrix = (squared.T @ squared) / n - sample_cov**2
    beta = float(np.sum(beta_matrix) / (n * p))
    beta = min(max(beta, 0.0), delta)
    shrinkage = 0.0 if delta <= np.finfo(float).eps else beta / delta
    covariance = (1.0 - shrinkage) * sample_cov + shrinkage * target

    diagnostics = validate_covariance(covariance, method=MethodName.LEDOIT_WOLF)
    assumptions = [
        "Rows are independent and identically distributed with finite fourth moments.",
        "The shrinkage target is trace(S)/p times the identity matrix.",
        "The shrinkage intensity is data-driven and clipped only to the theoretical [0, 1] interval.",
        "Ledoit-Wolf reduces variance in high dimensions but is not an outlier-resistant estimator.",
    ]
    warnings: list[str] = []
    if shrinkage == 0.0:
        warnings.append("Estimated shrinkage is zero, so singularity is not corrected.")
    return EstimationResult(
        method=MethodName.LEDOIT_WOLF,
        covariance=covariance,
        location=location.tolist(),
        weights=_unit_weights(x.shape[0], "Ledoit-Wolf regularization does not down-weight individual rows."),
        diagnostics=diagnostics,
        parameters={
            "divisor": "n_samples",
            "center": "arithmetic_mean",
            "column_scaling": "none",
            "target": "scalar_identity",
            "target_variance": target_variance,
            "shrinkage": shrinkage,
            "delta": delta,
            "beta": beta,
        },
        assumptions=assumptions,
        steps=_steps(
            started,
            [
                ("center", "Subtract each column's arithmetic mean."),
                ("sample_covariance", "Compute MLE covariance with divisor n_samples."),
                ("target", "Set target to trace(S)/p * I."),
                ("shrinkage", "Estimate Ledoit-Wolf intensity from independent finite-sample terms."),
                ("combine", "Form (1-alpha)S + alpha*F without spectrum clipping."),
                ("validate", "Independently check symmetry, PSD, SVD rank, and invertibility."),
            ],
        ),
        warnings=warnings,
    )


def _mad_scale(values: np.ndarray) -> np.ndarray:
    return np.median(np.abs(values - np.median(values, axis=0)), axis=0) / 0.6744897501960817


def _gk_pair_scale(value: np.ndarray) -> np.ndarray:
    return np.median(np.abs(value - np.median(value, axis=0)), axis=0) / 0.6744897501960817


def _ogk_step(
    z: np.ndarray, scale_tolerance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """One orthogonalized GK step preserving exact null spaces."""

    p = z.shape[1]
    scale = _mad_scale(z)
    active = scale > scale_tolerance
    if not np.any(active):
        raise EstimationError(
            "OGK requires at least one column with nonzero MAD scale.",
            stage="ogk_scale",
            details={"zero_scale_columns": list(range(p))},
        )

    standardized = np.zeros_like(z)
    standardized[:, active] = z[:, active] / scale[active]
    active_indices = np.flatnonzero(active)
    correlation = np.zeros((p, p), dtype=float)
    correlation[active_indices, active_indices] = 1.0
    for pos, i in enumerate(active_indices):
        for j in active_indices[:pos]:
            plus = _gk_pair_scale(standardized[:, i] + standardized[:, j])
            minus = _gk_pair_scale(standardized[:, i] - standardized[:, j])
            pair_covariance = float((plus**2 - minus**2) / 4.0)
            correlation[i, j] = pair_covariance
            correlation[j, i] = pair_covariance

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    transformed = standardized @ eigenvectors
    transform = np.diag(scale) @ eigenvectors
    warnings: list[str] = []
    if float(eigenvalues.min()) < -1e-10:
        warnings.append(
            "The intermediate pairwise GK correlation is indefinite; OGK orthogonalizes it. "
            "The final covariance is not produced by clipping its eigenvalues."
        )
    return scale, correlation, eigenvalues, transformed, transform, warnings


def ogk_covariance(
    sample: PreparedSample,
    *,
    reweight_quantile: float = 0.975,
    scale_tolerance: float = 1e-12,
) -> EstimationResult:
    """Two-step raw Orthogonalized Gnanadesikan-Kettenring covariance."""

    started = perf_counter()
    x = sample.values
    n, p = x.shape
    initial_center, initial_scale = robust_location_scale(x)
    centered = x - initial_center

    scale1, corr1, eig1, z1, transform1, warnings1 = _ogk_step(centered, scale_tolerance)
    scale2, corr2, eig2, z2, transform2, warnings2 = _ogk_step(z1, scale_tolerance)
    transform = transform1 @ transform2
    orthogonal_scale = _mad_scale(z2)
    orthogonal_center = np.median(z2, axis=0)
    raw_covariance = (transform * orthogonal_scale**2) @ transform.T
    raw_covariance = 0.5 * (raw_covariance + raw_covariance.T)
    distances = np.zeros(n, dtype=float)
    active = orthogonal_scale > scale_tolerance
    rank = int(np.count_nonzero(active))
    if rank > 0:
        distances = np.sum(
            ((z2[:, active] - orthogonal_center[active]) / orthogonal_scale[active]) ** 2,
            axis=1,
        )
        median_chi_square = float(stats.chi2.ppf(0.5, rank))
        scale_factor = float(np.median(distances) / median_chi_square)
    else:
        scale_factor = 1.0
    covariance = raw_covariance * scale_factor
    covariance = 0.5 * (covariance + covariance.T)
    location = initial_center + transform @ orthogonal_center

    diagnostics = validate_covariance(covariance, method=MethodName.OGK)
    if rank != diagnostics.rank:
        raise EstimationError(
            "OGK active-scale rank did not agree with the independently computed SVD rank.",
            stage="ogk_rank_consistency",
            details={"active_scale_rank": rank, "svd_rank": diagnostics.rank},
        )

    threshold = float(scale_factor * stats.chi2.ppf(reweight_quantile, rank)) if rank > 0 else float("inf")
    weights = [
        SampleWeight(
            index=i,
            weight=1.0 if float(distance) <= threshold else 0.0,
            robust_distance=float(distance),
            threshold=threshold,
            reason=(
                "Robust distance is within the chi-square hard-rejection cutoff."
                if float(distance) <= threshold
                else "Robust distance exceeds the chi-square hard-rejection cutoff."
            ),
        )
        for i, distance in enumerate(distances)
    ]

    assumptions = [
        "Rows are independent, and the clean bulk is approximately continuous and elliptical.",
        "Location and scale use medians and Gaussian-consistent MAD; the pairwise scatter is GK covariance.",
        "The returned scatter is the raw two-step OGK estimate; hard-rejection weights are diagnostics and do not change it.",
        "Columns with zero robust scale represent an exact null space rather than a repaired positive eigenvalue.",
        "A global scale factor aligns the median robust distance on the active rank with chi-square median calibration.",
    ]
    warnings = warnings1 + warnings2
    if n <= p:
        warnings.append(
            "n_samples <= n_features: OGK weights use an asymptotic chi-square cutoff and should be treated cautiously."
        )
    if not diagnostics.invertible:
        warnings.append("Exact or near-exact dependence makes the OGK matrix singular.")

    return EstimationResult(
        method=MethodName.OGK,
        covariance=covariance,
        location=location.tolist(),
        weights=weights,
        diagnostics=diagnostics,
        parameters={
            "iterations": 2,
            "location": "median",
            "scale": "Gaussian-consistent MAD",
            "pairwise_covariance": "Gnanadesikan-Kettenring",
            "column_scaling": "median/MAD standardization inside OGK, then explicit inverse scaling",
            "weight_rule": "hard chi-square rejection; diagnostic only for raw OGK",
            "weight_quantile": reweight_quantile,
            "weight_degrees_of_freedom": rank,
            "joint_scale_factor": scale_factor,
            "scale_tolerance": scale_tolerance,
        },
        assumptions=assumptions,
        steps=_steps(
            started,
            [
                ("robust_scale", "Compute per-column median and Gaussian-consistent MAD."),
                ("gk_step_1", "Form pairwise robust GK correlations and orthogonalize them."),
                ("gk_step_2", "Repeat MAD/GK orthogonalization while retaining exact null dimensions."),
                ("back_transform", "Transform orthogonal variances back and apply active-rank chi-square calibration."),
                ("weights", "Compute robust distances and calibrated hard chi-square anomaly weights without re-fitting."),
                ("validate", "Independently check symmetry, PSD, SVD rank, and invertibility."),
            ],
        ),
        warnings=warnings,
    )


ESTIMATORS: dict[MethodName, Callable[[PreparedSample], EstimationResult]] = {
    MethodName.EMPIRICAL: empirical_covariance,
    MethodName.LEDOIT_WOLF: ledoit_wolf_covariance,
    MethodName.OGK: ogk_covariance,
}
