"""Regularization-path orchestration with warm starts.

Responsibilities:
  * build or validate the lambda sequence (user sequences are solved in
    descending order for warm-start efficiency and mapped back to the
    requested order);
  * run the coordinate-descent solver along the path, threading the previous
    solution in as the next warm start;
  * aggregate per-lambda convergence into an overall decision:
    ACCEPTED (all converged), UNDETERMINED (some lambda did not converge),
    REJECTED (validation failed, raised by ``validation``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import diagnostics as diag
from .config import Settings
from .diagnostics import Decision, Diagnostic
from .preprocessing import lambda_max
from .solver import SolverResult, solve_lasso
from .validation import validate_fit_input


@dataclass
class LambdaSolution:
    lambda_: float
    coef: np.ndarray  # (p,) original scale, zeros on dropped columns
    intercept: float
    n_iter: int
    converged: bool
    duality_gap: float
    kkt_violation: float

    @property
    def support(self) -> np.ndarray:
        return np.flatnonzero(self.coef)


@dataclass
class PathResult:
    request_id: str
    decision: Decision
    solutions: list[LambdaSolution]  # in the requested lambda order
    diagnostics: list[Diagnostic] = field(default_factory=list)
    dropped_columns: list[int] = field(default_factory=list)


def _expand(coef_kept: np.ndarray, keep_idx: np.ndarray, p: int) -> np.ndarray:
    full = np.zeros(p)
    full[keep_idx] = coef_kept
    return full


def fit_path(
    X,
    y,
    *,
    lambdas=None,
    n_lambdas: int | None = None,
    lambda_min_ratio: float | None = None,
    fit_intercept: bool = True,
    tol: float | None = None,
    max_iter: int | None = None,
    zero_variance_policy: str = "reject",
    request_id: str,
    settings: Settings,
) -> PathResult:
    tol = settings.default_tol if tol is None else float(tol)
    max_iter = settings.default_max_iter if max_iter is None else int(max_iter)
    if tol <= 0 or max_iter < 1:
        raise ValueError("tol must be positive and max_iter >= 1")

    validated = validate_fit_input(
        X, y,
        lambdas=lambdas,
        fit_intercept=fit_intercept,
        zero_variance_policy=zero_variance_policy,
        zero_variance_eps=settings.zero_variance_eps,
        request_id=request_id,
    )
    Xv, yv, stats = validated.X, validated.y, validated.stats
    p_kept = Xv.shape[1]
    p_full = int(validated.keep_idx.max()) + 1 if validated.keep_idx.size else p_kept
    notes = list(validated.diagnostics)

    if lambdas is None:
        n_lam = settings.default_n_lambdas if n_lambdas is None else int(n_lambdas)
        ratio = settings.lambda_min_ratio if lambda_min_ratio is None else float(lambda_min_ratio)
        if not (1 <= n_lam <= settings.max_n_lambdas):
            raise ValueError(f"n_lambdas must be in [1, {settings.max_n_lambdas}]")
        if not (0.0 < ratio < 1.0):
            raise ValueError("lambda_min_ratio must be in (0, 1)")
        lmax = max(lambda_max(Xv, yv, stats), 1e-12)
        grid = np.logspace(np.log10(lmax), np.log10(lmax * ratio), n_lam)
        notes.append(diag.info(request_id, diag.LAMBDA_PATH_GENERATED,
                               "lambda path generated from data",
                               lambda_max=lmax, n_lambdas=int(n_lam),
                               lambda_min_ratio=ratio))
        requested = grid
        solve_order = np.arange(n_lam)
    else:
        requested = np.asarray(list(lambdas), dtype=np.float64)
        solve_order = np.argsort(-requested, kind="stable")
        if not np.all(np.diff(requested) <= 0):
            notes.append(diag.info(
                request_id, diag.LAMBDAS_REORDERED,
                "lambda sequence solved in descending order for warm starts; "
                "results are reported in the requested order",
                n_lambdas=int(requested.size)))

    results: list[tuple[int, SolverResult]] = []
    warm = None
    for pos in solve_order:
        res = solve_lasso(Xv, yv, stats, float(requested[pos]),
                          tol=tol, max_iter=max_iter, warm_start=warm)
        results.append((int(pos), res))
        warm = res.coef

    solutions: list[LambdaSolution] = [None] * len(results)  # type: ignore
    non_converged: list[dict] = []
    for pos, res in results:
        solutions[pos] = LambdaSolution(
            lambda_=float(requested[pos]),
            coef=_expand(res.coef, validated.keep_idx, p_full),
            intercept=res.intercept,
            n_iter=res.n_iter,
            converged=res.converged,
            duality_gap=res.duality_gap,
            kkt_violation=res.kkt_violation,
        )
        if not res.converged:
            non_converged.append(dict(lambda_=float(requested[pos]),
                                      n_iter=res.n_iter,
                                      duality_gap=res.duality_gap,
                                      kkt_violation=res.kkt_violation))

    if non_converged:
        notes.append(diag.warning(
            request_id, diag.NON_CONVERGENCE,
            "some lambdas did not meet the duality-gap criterion within "
            "max_iter; coefficients are returned but optimality is not "
            "certified",
            n_non_converged=len(non_converged),
            worst=[f"lambda={c['lambda_']:.6g} gap={c['duality_gap']:.3g}"
                   for c in non_converged[:3]]))
        decision = Decision.UNDETERMINED
    else:
        decision = Decision.ACCEPTED
        notes.append(diag.info(
            request_id, diag.PATH_COMPLETED, "all lambdas converged",
            n_lambdas=len(solutions),
            total_iter=sum(s.n_iter for s in solutions)))

    return PathResult(
        request_id=request_id,
        decision=decision,
        solutions=solutions,
        diagnostics=notes,
        dropped_columns=[int(i) for i in np.setdiff1d(np.arange(p_full), validated.keep_idx)],
    )
