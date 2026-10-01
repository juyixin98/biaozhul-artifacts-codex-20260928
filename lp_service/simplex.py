"""Deterministic two-phase simplex kernel.

The kernel operates on an equality model with nonnegative columns.  Pivoting
uses Bland's rule: the smallest eligible entering column and the smallest
column among ratio-test ties.  This is an explicit finite anti-cycling policy
for degenerate tableaux.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .parser import StandardLP


class KernelStatus(str, Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NUMERICAL = "numerical_uncertainty"


@dataclass
class SimplexResult:
    status: KernelStatus
    message: str
    x: np.ndarray
    reduced_costs: np.ndarray
    pi: np.ndarray | None
    phase1_q: np.ndarray | None
    active_rows: list[int]
    redundant_rows: list[int]
    iterations: int
    pivots: int
    pivot_log: list[dict[str, float | int | str]]
    entering: int | None = None
    standard_direction: np.ndarray | None = None
    artificial_remaining: int = 0


def _canonical_objective(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    basis: list[int],
) -> tuple[np.ndarray, float]:
    B = A[:, basis]
    c_b = c[basis]
    x_b = np.linalg.solve(B, b)
    pi = np.linalg.solve(B.T, c_b)
    d = c - A.T @ pi
    z = float(c_b @ x_b)
    return d, z


def _current_x(A: np.ndarray, b: np.ndarray, basis: list[int], ncols: int) -> np.ndarray:
    x = np.zeros(ncols, dtype=float)
    if basis:
        B = A[:, basis]
        x[basis] = np.linalg.solve(B, b)
    return x


def _pivot(
    tableau: np.ndarray,
    d: np.ndarray,
    z: float,
    basis: list[int],
    leaving_row: int,
    entering: int,
) -> float:
    pivot_value = tableau[leaving_row, entering]
    tableau[leaving_row, :] /= pivot_value
    pivot_row = tableau[leaving_row, :].copy()
    for row in range(tableau.shape[0]):
        if row == leaving_row:
            continue
        factor = tableau[row, entering]
        if factor != 0.0:
            tableau[row, :] -= factor * pivot_row

    objective_factor = d[entering]
    d -= objective_factor * pivot_row[:-1]
    # The tableau RHS stores -z. Row elimination therefore increases scalar z.
    z += objective_factor * pivot_row[-1]
    basis[leaving_row] = entering
    return z


def _bland_choice(
    tableau: np.ndarray,
    d: np.ndarray,
    basis: list[int],
    eligible: np.ndarray,
    zero_tol: float,
    pivot_tol: float,
) -> tuple[int | None, int | None, float | None]:
    candidates = np.flatnonzero((d < -zero_tol) & eligible)
    if candidates.size == 0:
        return None, None, None
    entering = int(candidates[0])
    column = tableau[:, entering]
    rhs = tableau[:, -1]
    ratios = np.full(tableau.shape[0], np.inf)
    valid = column > pivot_tol
    ratios[valid] = rhs[valid] / column[valid]
    valid &= ratios >= -zero_tol
    ratios[~valid] = np.inf
    finite = np.flatnonzero(np.isfinite(ratios))
    if finite.size == 0:
        return entering, None, None
    best_ratio = float(np.min(ratios[finite]))
    tie_rows = finite[np.abs(ratios[finite] - best_ratio) <= zero_tol * max(1.0, abs(best_ratio))]
    leaving_row = int(tie_rows[np.argmin(np.array(basis)[tie_rows])])
    return entering, leaving_row, best_ratio


def _simplex_phase(
    name: str,
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    basis: list[int],
    eligible: np.ndarray,
    max_iterations: int,
    zero_tol: float,
    pivot_tol: float,
    start_iterations: int,
    pivot_log: list[dict[str, float | int | str]],
) -> tuple[KernelStatus | None, np.ndarray, np.ndarray, float, list[int], int]:
    tableau = np.column_stack([A, b])
    d, z = _canonical_objective(A, b, c, basis)
    pivots = 0
    while True:
        x = _current_x(tableau[:, :-1], tableau[:, -1], basis, A.shape[1])
        entering, leaving_row, ratio = _bland_choice(
            tableau, d, basis, eligible, zero_tol, pivot_tol
        )
        if entering is None:
            return None, tableau, d, z, basis, pivots
        if leaving_row is None:
            direction = np.zeros(A.shape[1], dtype=float)
            direction[entering] = 1.0
            for row, basic_col in enumerate(basis):
                direction[basic_col] = -tableau[row, entering]
            pivot_log.append(
                {
                    "phase": name,
                    "iteration": start_iterations + pivots,
                    "entering": entering,
                    "leaving": -1,
                    "ratio": -1.0,
                    "objective_before": z,
                }
            )
            _ = x, ratio
            return KernelStatus.UNBOUNDED, tableau, d, z, basis, pivots

        pivot_log.append(
            {
                "phase": name,
                "iteration": start_iterations + pivots,
                "entering": int(entering),
                "leaving": int(basis[leaving_row]),
                "ratio": float(ratio if ratio is not None else -1.0),
                "objective_before": float(z),
            }
        )
        z = _pivot(tableau, d, z, basis, leaving_row, entering)
        z = float(c @ _current_x(tableau[:, :-1], tableau[:, -1], basis, A.shape[1]))
        pivots += 1
        if start_iterations + pivots > max_iterations:
            return KernelStatus.NUMERICAL, tableau, d, z, basis, pivots


def run_two_phase(
    problem: StandardLP,
    max_iterations: int = 1000,
    supplied_tolerance: float | None = None,
) -> SimplexResult:
    A0 = problem.A.copy()
    b0 = problem.b.copy()
    nrows, ncols = A0.shape
    scale = max(
        1.0,
        float(np.max(np.abs(A0))) if A0.size else 1.0,
        float(np.max(np.abs(b0))) if b0.size else 1.0,
    )
    zero_tol = supplied_tolerance or (1e-10 * scale)
    pivot_tol = max(1e-12, 1e-11 * scale)
    hard_tol = 100.0 * zero_tol
    pivot_log: list[dict[str, float | int | str]] = []

    artificial = np.array(list(problem.artificial_cols.values()), dtype=int)
    artificial_set = set(int(x) for x in artificial)
    eligible_phase1 = np.ones(ncols, dtype=bool)
    phase1_c = np.zeros(ncols, dtype=float)
    phase1_c[list(artificial_set)] = 1.0
    basis = list(problem.basis)

    active_rows = list(range(nrows))
    redundant_rows: list[int] = []
    q = None
    iterations = 0
    pivots = 0

    if nrows > 0:
        status, tableau, d, phase1_z, basis, pivots = _simplex_phase(
            "phase_1",
            A0,
            b0,
            phase1_c,
            basis,
            eligible_phase1,
            max_iterations,
            zero_tol,
            pivot_tol,
            0,
            pivot_log,
        )
        iterations = pivots
        if status == KernelStatus.UNBOUNDED:
            # Phase I is bounded below by zero; this only happens under severe
            # numerical corruption.
            return SimplexResult(
                KernelStatus.NUMERICAL,
                "phase I reported an impossible unbounded direction",
                _current_x(tableau[:, :-1], tableau[:, -1], basis, ncols),
                d,
                None,
                None,
                active_rows,
                redundant_rows,
                iterations,
                pivots,
                pivot_log,
            )
        if status == KernelStatus.NUMERICAL:
            return SimplexResult(
                status,
                "iteration limit reached in phase I",
                _current_x(tableau[:, :-1], tableau[:, -1], basis, ncols),
                d,
                None,
                None,
                active_rows,
                redundant_rows,
                iterations,
                pivots,
                pivot_log,
            )

        x_phase1 = _current_x(tableau[:, :-1], tableau[:, -1], basis, ncols)
        artificial_sum = float(np.sum(x_phase1[list(artificial_set)]))
        if artificial_sum > hard_tol:
            q_full = np.zeros(nrows, dtype=float)
            for row in range(nrows):
                art_col = problem.artificial_cols[row]
                q_full[row] = 1.0 - d[art_col]
            return SimplexResult(
                KernelStatus.INFEASIBLE,
                "phase I optimum has a positive artificial variable",
                x_phase1,
                d,
                None,
                q_full,
                active_rows,
                redundant_rows,
                iterations,
                pivots,
                pivot_log,
                artificial_remaining=int(
                    np.count_nonzero(x_phase1[list(artificial_set)] > hard_tol)
                ),
            )
        if artificial_sum > zero_tol:
            return SimplexResult(
                KernelStatus.NUMERICAL,
                "phase I artificial sum is near zero but above tolerance",
                x_phase1,
                d,
                None,
                None,
                active_rows,
                redundant_rows,
                iterations,
                pivots,
                pivot_log,
            )

        # Drive zero-valued artificial variables out of the basis.  If no
        # positive non-artificial coefficient is usable, the row is redundant.
        while True:
            basic_artificial_rows = [
                row for row, col in enumerate(basis) if col in artificial_set
            ]
            if not basic_artificial_rows:
                break
            changed = False
            for local_row in list(basic_artificial_rows):
                row_mask = np.array(
                    [
                        col not in artificial_set and tableau[local_row, col] > pivot_tol
                        for col in range(ncols)
                    ]
                )
                candidates = np.flatnonzero(row_mask)
                if candidates.size == 0:
                    standard_row = active_rows[local_row]
                    redundant_rows.append(standard_row)
                    tableau = np.delete(tableau, local_row, axis=0)
                    del basis[local_row]
                    active_rows.pop(local_row)
                    changed = True
                    break
                entering = int(candidates[0])
                leaving = basis[local_row]
                pivot_log.append(
                    {
                        "phase": "phase_1_artificial_exit",
                        "iteration": iterations,
                        "entering": entering,
                        "leaving": int(leaving),
                        "ratio": 0.0,
                        "objective_before": float(phase1_z),
                    }
                )
                phase1_z = _pivot(tableau, d, phase1_z, basis, local_row, entering)
                iterations += 1
                pivots += 1
                changed = True
                break
            if not changed:
                break

        A = tableau[:, :-1]
        b = tableau[:, -1]
    else:
        A = A0
        b = b0
        iterations = 0

    eligible_phase2 = np.ones(ncols, dtype=bool)
    eligible_phase2[list(artificial_set)] = False
    status, tableau, d, z, basis, phase2_pivots = _simplex_phase(
        "phase_2",
        A,
        b,
        problem.c,
        basis,
        eligible_phase2,
        max_iterations,
        zero_tol,
        pivot_tol,
        iterations,
        pivot_log,
    )
    iterations += phase2_pivots
    x = _current_x(tableau[:, :-1], tableau[:, -1], basis, ncols)
    x[np.abs(x) < zero_tol] = 0.0

    pi = None
    if nrows > 0:
        pi = np.zeros(nrows, dtype=float)
        for local_row, standard_row in enumerate(active_rows):
            art_col = problem.artificial_cols[standard_row]
            pi[standard_row] = -d[art_col]

    if status == KernelStatus.UNBOUNDED:
        last = pivot_log[-1]
        entering = int(last["entering"])
        direction = np.zeros(ncols, dtype=float)
        direction[entering] = 1.0
        for local_row, basic_col in enumerate(basis):
            direction[basic_col] = -tableau[local_row, entering]
        return SimplexResult(
            status,
            "phase II found a feasible improving ray",
            x,
            d,
            pi,
            None,
            active_rows,
            redundant_rows,
            iterations,
            pivots + phase2_pivots,
            pivot_log,
            entering=entering,
            standard_direction=direction,
        )
    if status == KernelStatus.NUMERICAL:
        return SimplexResult(
            status,
            "iteration limit reached in phase II",
            x,
            d,
            pi,
            None,
            active_rows,
            redundant_rows,
            iterations,
            pivots + phase2_pivots,
            pivot_log,
        )

    return SimplexResult(
        KernelStatus.OPTIMAL,
        "optimal basis found",
        x,
        d,
        pi,
        None,
        active_rows,
        redundant_rows,
        iterations,
        pivots + phase2_pivots,
        pivot_log,
    )
