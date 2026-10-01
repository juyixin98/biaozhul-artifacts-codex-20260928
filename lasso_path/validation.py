"""Input validation.

Policy layer on top of the numeric kernels in ``preprocessing``: checks
shapes, finiteness, lambda values, and the zero-variance-column contract
(reject by default, or drop-and-continue when explicitly requested).

Failures raise :class:`InputRejected` carrying diagnostics whose decision is
``REJECTED``; messages only contain shapes/indices/statistics, never raw
sample values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from . import diagnostics as diag
from .diagnostics import Diagnostic
from .preprocessing import ColumnStats, estimate_column_stats


class InputRejected(Exception):
    """Raised when validation rejects a request; carries the diagnostics."""

    def __init__(self, diagnostics: list[Diagnostic]):
        super().__init__(diagnostics[0].message if diagnostics else "input rejected")
        self.diagnostics = diagnostics


@dataclass
class ValidatedInput:
    X: object  # dense ndarray or scipy sparse matrix, kept columns only
    y: np.ndarray
    stats: ColumnStats
    keep_idx: np.ndarray  # indices of kept columns into the original design
    diagnostics: list[Diagnostic] = field(default_factory=list)


def _as_matrix(X, request_id: str, errors: list[Diagnostic]):
    if sparse.issparse(X):
        if not (sparse.isspmatrix_csc(X) or sparse.isspmatrix_csr(X)):
            X = X.tocsc()
        if X.data.size and not np.isfinite(X.data).all():
            errors.append(diag.error(request_id, diag.NON_FINITE_INPUT,
                                     "sparse matrix contains non-finite values",
                                     n_nonzero=int(X.nnz)))
        return X
    try:
        arr = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        errors.append(diag.error(request_id, diag.DIMENSION_MISMATCH,
                                 f"X is not a rectangular numeric matrix: {type(exc).__name__}"))
        return None
    if arr.ndim != 2:
        errors.append(diag.error(request_id, diag.DIMENSION_MISMATCH,
                                 "X must be 2-dimensional", ndim=int(arr.ndim)))
        return None
    if not np.isfinite(arr).all():
        errors.append(diag.error(request_id, diag.NON_FINITE_INPUT,
                                 "X contains NaN or infinite values",
                                 n_rows=int(arr.shape[0]), n_cols=int(arr.shape[1])))
    return arr


def validate_fit_input(
    X,
    y,
    *,
    lambdas,
    fit_intercept: bool,
    zero_variance_policy: str,
    zero_variance_eps: float,
    request_id: str,
) -> ValidatedInput:
    errors: list[Diagnostic] = []
    notes: list[Diagnostic] = []

    X = _as_matrix(X, request_id, errors)

    try:
        y_arr = np.asarray(y, dtype=np.float64).ravel()
    except (ValueError, TypeError):
        y_arr = None
        errors.append(diag.error(request_id, diag.DIMENSION_MISMATCH,
                                 "y is not a one-dimensional numeric vector"))
    if y_arr is not None and not np.isfinite(y_arr).all():
        errors.append(diag.error(request_id, diag.NON_FINITE_INPUT,
                                 "y contains NaN or infinite values",
                                 n_samples=int(y_arr.size)))

    if X is not None and y_arr is not None:
        n, p = X.shape
        if n < 2 or p < 1:
            errors.append(diag.error(request_id, diag.EMPTY_INPUT,
                                     "need at least 2 samples and 1 feature",
                                     n_samples=int(n), n_features=int(p)))
        if y_arr.shape[0] != n:
            errors.append(diag.error(
                request_id, diag.DIMENSION_MISMATCH,
                "X and y disagree on the number of samples",
                n_samples_X=int(n), n_samples_y=int(y_arr.shape[0])))

    if lambdas is not None:
        lam = np.asarray(list(lambdas), dtype=np.float64)
        if lam.size == 0:
            errors.append(diag.error(request_id, diag.INVALID_LAMBDA,
                                     "lambda sequence is empty"))
        elif not np.isfinite(lam).all() or (lam <= 0).any():
            errors.append(diag.error(
                request_id, diag.INVALID_LAMBDA,
                "every lambda must be finite and strictly positive",
                n_lambdas=int(lam.size)))

    if errors:
        raise InputRejected(errors)

    n, p = X.shape
    stats = estimate_column_stats(X, fit_intercept=fit_intercept)
    var = stats.scale ** 2
    zero_var = np.flatnonzero(var <= zero_variance_eps)

    keep_idx = np.arange(p)
    if zero_var.size:
        ctx = dict(n_columns=int(zero_var.size),
                   column_indices=[int(i) for i in zero_var])
        if zero_variance_policy == "reject":
            raise InputRejected([diag.error(
                request_id, diag.ZERO_VARIANCE_COLUMN,
                "constant (zero-variance) columns are rejected by policy; "
                "resend with zero_variance_policy='drop' to fit without them",
                **ctx)])
        if zero_variance_policy != "drop":
            raise InputRejected([diag.error(
                request_id, diag.INVALID_OPTION,
                "zero_variance_policy must be 'reject' or 'drop'",
                value=str(zero_variance_policy))])
        keep_idx = np.setdiff1d(np.arange(p), zero_var)
        notes.append(diag.info(
            request_id, diag.ZERO_VARIANCE_DROPPED,
            "constant columns dropped from the fit; their coefficients are 0",
            **ctx))
        X = X[:, keep_idx]
        stats = estimate_column_stats(X, fit_intercept=fit_intercept)

    notes.append(diag.info(
        request_id, diag.INPUT_VALIDATED, "input validated",
        n_samples=int(n), n_features=int(p), n_features_kept=int(keep_idx.size),
        sparse=bool(sparse.issparse(X)), fit_intercept=bool(fit_intercept)))

    return ValidatedInput(X=X, y=y_arr, stats=stats, keep_idx=keep_idx,
                          diagnostics=notes)
