from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class MatrixDiagnostics:
    symmetric: bool
    positive_semidefinite: bool
    positive_definite: bool
    symmetry_error: float
    min_eigenvalue: float
    max_eigenvalue: float
    eigenvalues: np.ndarray
    rank: int
    invertible: bool
    condition_number: float | None
    validation_failures: list[str]


@dataclass(frozen=True)
class EstimatorResult:
    method: str
    covariance: np.ndarray
    shrinkage: float
    weights: np.ndarray
    iterations: int
    converged: bool
    assumptions: list[str]
    steps: list[dict[str, object]]
    failures: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    diagnostics: MatrixDiagnostics | None = None


def diagnose_matrix(matrix: np.ndarray) -> MatrixDiagnostics:
    """Check symmetry and PSD independently; no eigenvalue clipping occurs."""
    m = np.asarray(matrix, dtype=float)
    failures: list[str] = []
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        failures.append("MATRIX_SHAPE: result is not a square matrix")
        p = m.shape[0] if m.ndim == 2 else 0
        empty = np.full(max(p, 0), np.nan)
        return MatrixDiagnostics(
            False, False, False, np.inf, np.nan, np.nan, empty, 0, False, None, failures
        )

    symmetry_error = float(np.max(np.abs(m - m.T))) if m.size else 0.0
    scale = float(np.max(np.abs(m))) if m.size else 0.0
    symmetry_tol = max(1e-10, 1e-9 * scale)
    symmetric = bool(np.isfinite(symmetry_error) and symmetry_error <= symmetry_tol)
    if not symmetric:
        failures.append(
            f"SYMMETRY: max |C-C^T|={symmetry_error:.12g} exceeds {symmetry_tol:.12g}"
        )

    if symmetric:
        eigenvalues = np.linalg.eigvalsh(m)
    else:
        # This is a separate PSD check on the unmodified result. It is not
        # symmetrized first, because that could hide an algorithm defect.
        eigenvalues = np.linalg.eigvals(m).real
        eigenvalues.sort()

    finite = bool(np.all(np.isfinite(eigenvalues)))
    min_eig = float(eigenvalues[0]) if finite else float("nan")
    max_eig = float(eigenvalues[-1]) if finite else float("nan")
    tol = max(m.shape) * np.finfo(float).eps * max(0.0, max_eig)
    psd = bool(finite and min_eig >= -tol)
    pd = bool(finite and min_eig > tol)
    rank = int(np.count_nonzero(eigenvalues > tol)) if finite else 0
    invertible = pd and rank == m.shape[0]
    condition = float(max_eig / min_eig) if invertible and min_eig > 0.0 else None

    if not finite:
        failures.append("FINITE_MATRIX: result contains non-finite eigenvalues")
    if not psd:
        failures.append(
            f"POSITIVE_SEMIDEFINITE: minimum eigenvalue {min_eig:.12g} is negative"
        )

    return MatrixDiagnostics(
        symmetric=symmetric,
        positive_semidefinite=psd,
        positive_definite=pd,
        symmetry_error=symmetry_error,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
        eigenvalues=eigenvalues,
        rank=rank,
        invertible=invertible,
        condition_number=condition,
        validation_failures=failures,
    )


def _step(name: str, started: float, **details: object) -> dict[str, object]:
    event = {"step": name, "elapsed_ms": round((perf_counter() - started) * 1000.0, 4)}
    event.update(details)
    return event


def ledoit_wolf_shrinkage(z: np.ndarray) -> EstimatorResult:
    """Ledoit-Wolf shrinkage toward mean-variance diagonal covariance.

    The sample covariance uses denominator n. This is the non-robust
    comparison method and is useful when rows are approximately i.i.d.
    elliptical/Gaussian without severe contamination.
    """
    started = perf_counter()
    n, p = z.shape
    sample_cov = (z.T @ z) / n
    target_scale = float(np.trace(sample_cov) / p)
    target = target_scale * np.eye(p)

    squared = z * z
    phi_matrix = (squared.T @ squared) / n - sample_cov * sample_cov
    beta_numerator = float(np.sum(phi_matrix)) / n
    delta = float(np.sum((sample_cov - target) ** 2) / p)
    raw_shrinkage = beta_numerator / delta if delta > np.finfo(float).eps else 0.0
    shrinkage = float(np.clip(raw_shrinkage, 0.0, 1.0))
    covariance = (1.0 - shrinkage) * sample_cov + shrinkage * target
    diagnostics = diagnose_matrix(covariance)

    assumptions = [
        "Rows are independent and identically distributed observations.",
        "The estimator is non-robust: strong outliers can move location and scatter.",
        "Shrinkage target is diag(mean sample variance * I); covariance uses denominator n.",
    ]
    steps = [
        _step("ledoit_wolf.sample_covariance", started, denominator=n),
        _step(
            "ledoit_wolf.shrinkage",
            started,
            raw_shrinkage=raw_shrinkage,
            applied_shrinkage=shrinkage,
            target_diagonal=target_scale,
        ),
        _step(
            "matrix.validate_independent_checks",
            started,
            symmetric=diagnostics.symmetric,
            positive_semidefinite=diagnostics.positive_semidefinite,
            rank=diagnostics.rank,
        ),
    ]
    uncertainties: list[str] = []
    if target_scale == 0.0:
        uncertainties.append(
            "ZERO_VARIANCE: all standardized residual variance is zero; covariance cannot be invertible"
        )
    if not np.isfinite(raw_shrinkage) or raw_shrinkage != shrinkage:
        uncertainties.append(
            f"SHRINKAGE_CLIPPED: raw Ledoit-Wolf value {raw_shrinkage:.12g} was clipped to [0,1]"
        )

    return EstimatorResult(
        method="ledoit_wolf",
        covariance=covariance,
        shrinkage=shrinkage,
        weights=np.ones(n),
        iterations=1,
        converged=True,
        assumptions=assumptions,
        steps=steps,
        uncertainties=uncertainties,
        diagnostics=diagnostics,
    )


def _tyler_plugin_shrinkage(z: np.ndarray, scale_mad: np.ndarray) -> tuple[float, float]:
    """Chen-Wiesel plugin shrinkage used by the robust Tyler iteration."""
    n, p = z.shape
    sample_std = z.std(axis=0, ddof=0)
    safe_std = np.where(sample_std > np.finfo(float).eps, sample_std, 1.0)
    pilot = (z / safe_std).T @ (z / safe_std)
    pilot *= np.outer(scale_mad, scale_mad)
    trace = float(np.trace(pilot))
    if trace <= np.finfo(float).eps:
        return 1.0, 1.0
    pilot *= p / trace
    tr = float(np.trace(pilot @ pilot))
    denominator = (p * p - n * p - 2.0 * n) + (
        n + 1.0 + 2.0 * (n - 1.0) / p
    ) * tr
    raw = (p * p + (1.0 - 2.0 / p) * tr) / denominator
    return raw, float(np.clip(raw, 0.0, 1.0))


def robust_tyler_shrinkage(
    z: np.ndarray,
    *,
    shrinkage: float | None = None,
    max_iterations: int = 200,
    tolerance: float = 1e-9,
) -> EstimatorResult:
    """Regularized Tyler M-estimator with an explicit fixed-point iteration.

    Each observation receives weight ``p / (x_i^T C^{-1} x_i)``; a small
    weight therefore identifies a remote elliptical observation. The fixed
    point is mixed with the identity shape and trace-normalized. This gives
    an invertible high-dimensional estimate whenever positive shrinkage is
    selected, while reporting non-convergence instead of hiding it.
    """
    started = perf_counter()
    n, p = z.shape
    uncertainties: list[str] = []
    steps: list[dict[str, object]] = []

    # MAD is measured around zero because Tyler models centered residuals.
    scale_mad = np.median(np.abs(z), axis=0) / 0.6745
    zero_mad_columns = np.where(scale_mad <= np.finfo(float).eps)[0]
    initial_scale = np.where(scale_mad > np.finfo(float).eps, scale_mad, 1.0) ** 2
    cov = np.diag(initial_scale)

    raw_shrinkage = shrinkage
    if shrinkage is None:
        raw_shrinkage, shrinkage = _tyler_plugin_shrinkage(z, scale_mad)
    assert shrinkage is not None
    if raw_shrinkage is not None and not (0.0 <= raw_shrinkage <= 1.0):
        uncertainties.append(
            f"SHRINKAGE_CLIPPED: robust plugin value {float(raw_shrinkage):.12g} was clipped to {shrinkage:.12g}"
        )
    if zero_mad_columns.size:
        uncertainties.append(
            f"ZERO_MAD_COLUMNS: columns {zero_mad_columns.tolist()} have zero median absolute residual"
        )
    steps.append(
        _step(
            "tyler.initialize",
            started,
            raw_shrinkage=None if raw_shrinkage is None else float(raw_shrinkage),
            applied_shrinkage=shrinkage,
            zero_mad_columns=zero_mad_columns.tolist(),
        )
    )

    converged = False
    iterations = 0
    final_distances: np.ndarray | None = None
    for iterations in range(1, max_iterations + 1):
        old = cov
        inverse = np.linalg.pinv(old)
        distances = np.einsum("ij,jk,ik->i", z, inverse, z, optimize=True)
        positive = distances > np.finfo(float).eps
        safe_distances = np.where(positive, distances, 1.0)
        weights = np.where(positive, p / safe_distances, 0.0)
        weighted = np.sqrt(weights)[:, np.newaxis] * z
        unregularized = (weighted.T @ weighted) / n
        cov = (1.0 - shrinkage) * unregularized + shrinkage * np.eye(p)
        trace = float(np.trace(cov))
        if trace > np.finfo(float).eps:
            cov *= p / trace
        final_distances = distances

        difference = float(np.max(np.abs(cov - old)))
        scale_old = float(np.max(np.abs(old)))
        relative_change = difference / max(1.0, scale_old)
        if relative_change <= tolerance:
            converged = True
            break

    if final_distances is None:
        raise RuntimeError("Tyler iteration did not execute")
    positive = final_distances > np.finfo(float).eps
    safe_distances = np.where(positive, final_distances, 1.0)
    weights = np.where(positive, p / safe_distances, 0.0)
    zero_rows = np.where(~positive)[0]
    if zero_rows.size:
        uncertainties.append(
            f"ZERO_RESIDUAL_ROWS: rows {zero_rows.tolist()} receive zero scatter weight"
        )

    diagnostics = diagnose_matrix(cov)
    steps.append(
        _step(
            "tyler.fixed_point",
            started,
            iterations=iterations,
            converged=converged,
            final_relative_change=relative_change,
            tolerance=tolerance,
        )
    )
    steps.append(
        _step(
            "matrix.validate_independent_checks",
            started,
            symmetric=diagnostics.symmetric,
            positive_semidefinite=diagnostics.positive_semidefinite,
            rank=diagnostics.rank,
        )
    )
    failures: list[str] = []
    if not converged:
        failures.append(
            f"NON_CONVERGED: Tyler fixed point did not converge within {max_iterations} iterations"
        )
    failures.extend(diagnostics.validation_failures)

    assumptions = [
        "Rows are independent observations from an approximately elliptical distribution.",
        "Tyler's M-estimator identifies scatter up to scale; trace normalization fixes the scale.",
        "Outliers receive low weights p / squared Mahalanobis distance.",
        "The estimator is robust to radial contamination, not to arbitrary structural dependence changes.",
    ]
    return EstimatorResult(
        method="robust_tyler_shrinkage",
        covariance=cov,
        shrinkage=float(shrinkage),
        weights=weights,
        iterations=iterations,
        converged=converged,
        assumptions=assumptions,
        steps=steps,
        failures=failures,
        uncertainties=uncertainties,
        diagnostics=diagnostics,
    )
