"""Model parsing: validate a ProblemSpec into a CanonicalProblem.

All semantic validation of the optimization model happens here and raises
categorized errors (InputValidationError / ResourceExhaustedError) so the
same checks apply to HTTP and in-process callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import SolverConfig
from .errors import InputValidationError, ResourceExhaustedError
from .schemas import ProblemSpec


@dataclass
class CanonicalProblem:
    c: np.ndarray           # (n,)
    A_ub: np.ndarray        # (m_ub, n), possibly 0 rows
    b_ub: np.ndarray        # (m_ub,)
    A_eq: np.ndarray        # (m_eq, n), possibly 0 rows
    b_eq: np.ndarray        # (m_eq,)
    lb: np.ndarray          # (n,), may contain -inf
    ub: np.ndarray          # (n,), may contain +inf
    integer_idx: np.ndarray  # (k,) indices of integer variables
    maximize: bool
    tol_scale: float        # scales absolute feasibility tolerances
    n: int


def _check_finite(name: str, values: list[float]) -> None:
    for i, v in enumerate(values):
        if not math.isfinite(v):
            raise InputValidationError(
                f"{name} contains a non-finite value at flat index {i}: {v!r}",
                {"field": name, "index": i},
            )


def _parse_constraints(a_name: str, b_name: str,
                       A: list[list[float]] | None, b: list[float] | None,
                       n: int) -> tuple[np.ndarray, np.ndarray]:
    if A is None and b is None:
        return np.zeros((0, n)), np.zeros(0)
    if A is None or b is None:
        raise InputValidationError(
            f"{a_name} and {b_name} must be provided together",
            {"field": a_name if A is None else b_name},
        )
    if len(A) != len(b):
        raise InputValidationError(
            f"{a_name} has {len(A)} rows but {b_name} has length {len(b)}",
            {"field": a_name},
        )
    for r, row in enumerate(A):
        if len(row) != n:
            raise InputValidationError(
                f"{a_name}[{r}] has length {len(row)}; expected {n} (one per variable)",
                {"field": a_name, "row": r},
            )
        _check_finite(f"{a_name}[{r}]", row)
    _check_finite(b_name, list(b))
    return np.asarray(A, dtype=float).reshape(len(A), n), np.asarray(b, dtype=float)


def parse_problem(spec: ProblemSpec, config: SolverConfig) -> CanonicalProblem:
    c_list = list(spec.c)
    if not c_list:
        raise InputValidationError("objective vector c must be non-empty")
    n = len(c_list)
    if n > config.max_variables:
        raise ResourceExhaustedError(
            f"problem has {n} variables; server limit is {config.max_variables}",
            {"variables": n, "limit": config.max_variables},
        )
    _check_finite("c", c_list)

    integrality = list(spec.integrality) if spec.integrality is not None else [1] * n
    if len(integrality) != n:
        raise InputValidationError(
            f"integrality has length {len(integrality)} but c has length {n}",
            {"field": "integrality"},
        )
    for i, v in enumerate(integrality):
        if v not in (0, 1):
            raise InputValidationError(
                f"integrality[{i}] must be 0 (continuous) or 1 (integer), got {v!r}",
                {"field": "integrality", "index": i},
            )

    bounds = list(spec.bounds) if spec.bounds is not None else [(0.0, None)] * n
    if len(bounds) != n:
        raise InputValidationError(
            f"bounds has length {len(bounds)} but c has length {n}",
            {"field": "bounds"},
        )
    lb = np.empty(n)
    ub = np.empty(n)
    for i, (lo, hi) in enumerate(bounds):
        lo_v = -math.inf if lo is None else float(lo)
        hi_v = math.inf if hi is None else float(hi)
        if math.isnan(lo_v) or math.isnan(hi_v):
            raise InputValidationError(
                f"bounds[{i}] contains NaN", {"field": "bounds", "index": i}
            )
        if lo_v > hi_v:
            raise InputValidationError(
                f"bounds[{i}]: lower bound {lo_v} exceeds upper bound {hi_v}",
                {"field": "bounds", "index": i},
            )
        lb[i], ub[i] = lo_v, hi_v

    integer_idx = np.array([i for i, v in enumerate(integrality) if v == 1], dtype=int)
    for i in integer_idx:
        if not (np.isfinite(lb[i]) and np.isfinite(ub[i])):
            raise InputValidationError(
                f"integer variable x[{int(i)}] must have finite lower and upper "
                "bounds (bounded integer program)",
                {"field": "bounds", "index": int(i)},
            )

    A_ub, b_ub = _parse_constraints("A_ub", "b_ub", spec.A_ub, spec.b_ub, n)
    A_eq, b_eq = _parse_constraints("A_eq", "b_eq", spec.A_eq, spec.b_eq, n)
    m = A_ub.shape[0] + A_eq.shape[0]
    if m > config.max_constraints:
        raise ResourceExhaustedError(
            f"problem has {m} constraints; server limit is {config.max_constraints}",
            {"constraints": m, "limit": config.max_constraints},
        )

    finite_magnitudes = [1.0]
    if b_ub.size:
        finite_magnitudes.append(float(np.max(np.abs(b_ub))))
    if b_eq.size:
        finite_magnitudes.append(float(np.max(np.abs(b_eq))))
    finite_lb = lb[np.isfinite(lb)]
    finite_ub = ub[np.isfinite(ub)]
    if finite_lb.size:
        finite_magnitudes.append(float(np.max(np.abs(finite_lb))))
    if finite_ub.size:
        finite_magnitudes.append(float(np.max(np.abs(finite_ub))))
    tol_scale = max(finite_magnitudes)

    return CanonicalProblem(
        c=np.asarray(c_list, dtype=float),
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        lb=lb, ub=ub,
        integer_idx=integer_idx,
        maximize=(spec.sense == "maximize"),
        tol_scale=tol_scale,
        n=n,
    )
