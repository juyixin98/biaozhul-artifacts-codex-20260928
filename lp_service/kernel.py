"""Dense two-phase simplex implementation.

Anti-cycling uses Bland's rule for every entering and leaving choice. An
iteration cap is a separate numerical guard; degeneracy itself is handled by
the deterministic index tie breaks.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import SolverConfig
from .standardize import RowMeta, StandardModel


@dataclass
class KernelResult:
    status: str
    standard_solution: np.ndarray | None = None
    standard_direction: np.ndarray | None = None
    equality_duals: np.ndarray | None = None
    reduced_costs: np.ndarray | None = None
    objective_value: float | None = None
    iterations: int = 0
    active_rows: np.ndarray | None = None
    final_basis: np.ndarray | None = None
    tableau: np.ndarray | None = None
    history: list[dict[str, object]] = field(default_factory=list)


def two_phase_simplex(
    model: StandardModel,
    config: SolverConfig = SolverConfig(),
    request_id: str = "",
) -> KernelResult:
    history: list[dict[str, object]] = []
    n = model.A.shape[1]
    if model.A.shape[0] == 0:
        return _no_rows_result(model, config, history)

    rows = list(model.rows)
    m = len(rows)
    basis = np.asarray([meta.initial_basic for meta in rows], dtype=int)
    active_rows = np.ones(m, dtype=bool)
    tableau = np.vstack(
        [
            np.hstack([model.A, model.d.reshape(-1, 1)]),
            np.zeros((1, n + 1)),
        ]
    )
    artificial_cols = np.arange(model.core_column_count + m, model.core_column_count + 2 * m)
    tableau[-1, :] = _initial_phase_one_row(tableau[:-1, :], artificial_cols)

    iterations = 0
    for phase in (1, 2):
        while True:
            rhs_min = float(np.min(tableau[:-1, -1]))
            if rhs_min < -config.uncertain_tol:
                return _uncertain_result(
                    model, tableau, basis, active_rows, iterations, history,
                    f"{request_id}: negative RHS {rhs_min:.3e} after pivots", phase,
                )

            eligible = np.arange(n)
            if phase == 2:
                eligible = eligible[eligible < model.core_column_count + m]
            reduced_costs = tableau[-1, :n]
            candidates = eligible[reduced_costs[eligible] < -config.zero_tol]
            if candidates.size == 0:
                phase_one_objective = -tableau[-1, -1]
                if phase == 1 and phase_one_objective > config.uncertain_tol:
                    y = _phase_one_duals(tableau, rows, model)
                    return KernelResult(
                        status="infeasible",
                        equality_duals=y,
                        reduced_costs=reduced_costs.copy(),
                        objective_value=float(tableau[-1, -1]),
                        iterations=iterations,
                        active_rows=active_rows,
                        final_basis=basis,
                        tableau=tableau,
                        history=history,
                    )
                if phase == 1 and tableau[-1, -1] > config.zero_tol:
                    return _uncertain_result(
                        model, tableau, basis, active_rows, iterations, history,
                        (
                            f"{request_id}: Phase-I objective {phase_one_objective:.3e} "
                            "is between the zero and uncertainty tolerances"
                        ),
                        phase,
                    )
                break

            entering = int(candidates[0])
            leaving, ratio = _bland_leaving_row(tableau, basis, entering, config)
            if leaving is None:
                if phase == 2:
                    direction = _ray(tableau, basis, entering, n)
                    return KernelResult(
                        status="unbounded",
                        standard_solution=_standard_solution(tableau, basis, n),
                        standard_direction=direction,
                        iterations=iterations,
                        active_rows=active_rows,
                        final_basis=basis,
                        tableau=tableau,
                        history=history,
                    )
                return _uncertain_result(
                    model, tableau, basis, active_rows, iterations, history,
                    f"{request_id}: Phase-I column {entering} has no positive pivot",
                    phase,
                )

            history.append(
                {
                    "phase": phase,
                    "iteration": iterations,
                    "entering": entering,
                    "leaving_row": int(leaving),
                    "leaving_variable": int(basis[leaving]),
                    "ratio": float(ratio),
                    "objective": float(tableau[-1, -1]),
                }
            )
            tableau, basis = _pivot(tableau, basis, leaving, entering)
            iterations += 1
            if iterations > config.max_iterations:
                return _uncertain_result(
                    model, tableau, basis, active_rows, iterations, history,
                    f"{request_id}: exceeded {config.max_iterations} simplex iterations",
                    phase,
                )

        if phase == 1:
            result = _leave_phase_one(
                tableau, basis, active_rows, rows, model, config, request_id, history,
            )
            if result is not None:
                return result
            tableau[-1, :] = _phase_two_objective_row(tableau[:-1, :], model, basis)

    y = _phase_two_duals(model, basis)
    return KernelResult(
        status="optimal",
        standard_solution=_standard_solution(tableau, basis, n),
        equality_duals=y,
        reduced_costs=tableau[-1, :n].copy(),
        objective_value=float(tableau[-1, -1]),
        iterations=iterations,
        active_rows=active_rows,
        final_basis=basis,
        tableau=tableau,
        history=history,
    )


def _initial_phase_one_row(A_with_rhs: np.ndarray, artificial_cols: np.ndarray) -> np.ndarray:
    row = np.zeros(A_with_rhs.shape[1])
    for col in artificial_cols:
        row[col] = 1.0
    row[: A_with_rhs.shape[1] - 1] -= np.sum(A_with_rhs[:, :-1], axis=0)
    row[-1] = -float(np.sum(A_with_rhs[:, -1]))
    return row


def _bland_leaving_row(
    tableau: np.ndarray,
    basis: np.ndarray,
    entering: int,
    config: SolverConfig,
) -> tuple[int | None, float]:
    column = tableau[:-1, entering]
    positive = np.flatnonzero(column > config.pivot_tol)
    if positive.size == 0:
        return None, np.inf
    ratios = tableau[positive, -1] / column[positive]
    best = float(np.min(ratios))
    near_best = positive[np.abs(ratios - best) <= config.zero_tol]
    leaving = int(near_best[np.argmin(basis[near_best])])
    return leaving, best


def _pivot(tableau: np.ndarray, basis: np.ndarray, leaving: int, entering: int):
    tableau = tableau.copy()
    basis = basis.copy()
    pivot_value = tableau[leaving, entering]
    tableau[leaving, :] /= pivot_value
    for r in range(tableau.shape[0]):
        if r == leaving:
            continue
        factor = tableau[r, entering]
        if factor != 0.0:
            tableau[r, :] -= factor * tableau[leaving, :]
    basis[leaving] = entering
    return tableau, basis


def _leave_phase_one(
    tableau: np.ndarray,
    basis: np.ndarray,
    active_rows: np.ndarray,
    rows: list[RowMeta],
    model: StandardModel,
    config: SolverConfig,
    request_id: str,
    history: list[dict[str, object]],
) -> KernelResult | None:
    m = model.A.shape[0]
    artificial = set(range(model.core_column_count + m, model.core_column_count + 2 * m))
    delete_rows: list[int] = []

    # Artificial basic variables at value zero must either leave or identify a
    # redundant equality. Pivoting on a nonzero non-artificial entry is exact
    # algebraically because its RHS is zero.
    for r in range(m):
        if basis[r] not in artificial:
            continue
        non_artificial_limit = model.core_column_count + m
        choices = np.flatnonzero(np.abs(tableau[r, :non_artificial_limit]) > config.pivot_tol)
        if choices.size:
            tableau, basis = _pivot(tableau, basis, r, int(choices[0]))
        else:
            delete_rows.append(r)

    if not delete_rows:
        return None

    keep = np.ones(m, dtype=bool)
    keep[np.asarray(delete_rows, dtype=int)] = False
    tableau = np.vstack([tableau[:-1, :][keep, :], tableau[-1, :]])
    basis = basis[keep]
    active_rows[:] = active_rows & keep

    new_index = 0
    for meta, is_active in zip(rows, keep):
        if is_active:
            object.__setattr__(meta, "row_index", new_index)
            new_index += 1
        else:
            object.__setattr__(meta, "row_index", -1)

    history.append(
        {
            "event": "redundant_rows_removed",
            "rows": delete_rows,
            "request_detail": f"{request_id}: zero artificial basic rows",
        }
    )
    return None


def _phase_two_objective_row(A_with_rhs: np.ndarray, model: StandardModel, basis: np.ndarray) -> np.ndarray:
    row = np.zeros(A_with_rhs.shape[1])
    row[: model.c.size] = model.c
    for r, basic_col in enumerate(basis):
        row -= model.c[basic_col] * A_with_rhs[r]
    return row


def _standard_solution(tableau: np.ndarray, basis: np.ndarray, n: int) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    for r, basic_col in enumerate(basis):
        if basic_col < n:
            x[basic_col] = max(0.0, float(tableau[r, -1]))
    return x


def _ray(tableau: np.ndarray, basis: np.ndarray, entering: int, n: int) -> np.ndarray:
    direction = np.zeros(n, dtype=float)
    direction[entering] = 1.0
    for r, basic_col in enumerate(basis):
        if basic_col < n:
            direction[basic_col] = -float(tableau[r, entering])
    return direction


def _phase_one_duals(
    tableau: np.ndarray,
    rows: list[RowMeta],
    model: StandardModel,
) -> np.ndarray:
    y = np.zeros(len(rows), dtype=float)
    m = len(rows)
    for i, meta in enumerate(rows):
        initial_col = meta.initial_basic
        initial_cost = 1.0 if initial_col >= model.core_column_count + m else 0.0
        y[i] = initial_cost + tableau[-1, initial_col]
    return y


def _phase_two_duals(model: StandardModel, basis: np.ndarray) -> np.ndarray:
    m = model.A.shape[0]
    # Final basic columns in original tableau coordinates give B. For a basic
    # solution B^{-1} is represented implicitly, and B^T y = c_B gives y.
    B = model.A[np.ix_(np.arange(m), basis)]
    cB = model.c[basis]
    try:
        return np.linalg.solve(B.T, cB)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(B.T, cB, rcond=None)[0]


def _no_rows_result(
    model: StandardModel,
    config: SolverConfig,
    history: list[dict[str, object]],
) -> KernelResult:
    # Standard model contains only free non-negative variables. If any c_j < 0,
    # moving along that non-negative ray is unbounded for minimization.
    candidates = np.flatnonzero(model.c < -config.zero_tol)
    if candidates.size == 0:
        n = model.c.size
        return KernelResult(
            status="optimal",
            standard_solution=np.zeros(n, dtype=float),
            equality_duals=np.zeros(0, dtype=float),
            reduced_costs=model.c.copy(),
            objective_value=0.0,
            active_rows=np.ones(0, dtype=bool),
            final_basis=np.empty(0, dtype=int),
            history=history,
        )

    direction = np.zeros(model.c.size, dtype=float)
    direction[int(candidates[0])] = 1.0
    return KernelResult(
        status="unbounded",
        standard_solution=np.zeros(model.c.size, dtype=float),
        standard_direction=direction,
        active_rows=np.ones(0, dtype=bool),
        final_basis=np.empty(0, dtype=int),
        history=history,
    )


def _uncertain_result(
    model: StandardModel,
    tableau: np.ndarray,
    basis: np.ndarray,
    active_rows: np.ndarray,
    iterations: int,
    history: list[dict[str, object]],
    reason: str,
    phase: int,
) -> KernelResult:
    history.append({"event": "numerical_uncertain", "phase": phase, "reason": reason})
    n = model.A.shape[1]
    return KernelResult(
        status="numerical_uncertain",
        standard_solution=_standard_solution(tableau, basis, n),
        iterations=iterations,
        active_rows=active_rows.copy(),
        final_basis=basis.copy(),
        tableau=tableau.copy(),
        history=history,
    )
