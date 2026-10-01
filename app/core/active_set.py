import numpy as np

from app.config import Tolerances
from app.core.feasibility import find_feasible_point, violation
from app.core.linalg import is_row_independent, rank_and_basis, solve_face_kkt
from app.core.types import ActiveSetResult, SolverEvent
from app.errors import NumericalError
from app.parser import ParsedProblem


def _event(iteration: int, phase: str, kind: str, **detail) -> SolverEvent:
    return SolverEvent(iteration=iteration, phase=phase, type=kind, detail=detail)


def _preprocess_equalities(
    problem: ParsedProblem, tol: Tolerances, events: list[SolverEvent]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if problem.n_equalities == 0:
        return (
            np.empty((0, problem.n)),
            np.empty(0),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
        )
    rank, basis_indices, _, dependent_indices = rank_and_basis(problem.A, tol.rank)
    events.append(
        _event(
            0,
            "preprocess",
            "equality_rank_detection",
            equality_rows=problem.n_equalities,
            rank=rank,
            basis_indices=basis_indices.tolist(),
            dependent_indices=dependent_indices.tolist(),
            rank_tol=tol.rank,
        )
    )
    A_basis = problem.A[basis_indices, :] if basis_indices.size else np.empty((0, problem.n))
    b_basis = problem.b[basis_indices] if basis_indices.size else np.empty(0)
    if dependent_indices.size:
        events.append(
            _event(
                0,
                "preprocess",
                "dependent_equalities_deferred_for_residual_check",
                dependent_indices=dependent_indices.tolist(),
            )
        )
    return A_basis, b_basis, basis_indices, dependent_indices


def _check_dependent_equality_consistency(
    x: np.ndarray,
    problem: ParsedProblem,
    dependent_indices: np.ndarray,
    tol: Tolerances,
) -> float:
    if dependent_indices.size == 0:
        return 0.0
    return float(np.max(np.abs(problem.A[dependent_indices, :] @ x - problem.b[dependent_indices])))


def _initial_active(
    x: np.ndarray, problem: ParsedProblem, tol: Tolerances, events: list[SolverEvent]
) -> list[int]:
    slack = problem.h - problem.G @ x if problem.n_inequalities else np.asarray([])
    candidates = [i for i, value in enumerate(slack) if abs(float(value)) <= tol.active]
    active: list[int] = []
    for index in candidates:
        if is_row_independent(
            np.vstack(
                [problem.A, problem.G[active, :]]
                if active
                else [problem.A]
            ),
            problem.G[index],
            tol.rank,
        ):
            active.append(index)
        else:
            events.append(
                _event(
                    0,
                    "initialization",
                    "dependent_active_inequality_skipped",
                    constraint_index=index,
                    constraint_name=problem.inequality_names[index],
                    slack=float(slack[index]),
                )
            )
    events.append(
        _event(
            0,
            "initialization",
            "initial_active_set",
            active_inequality_indices=active,
            candidate_slacks={str(i): float(slack[i]) for i in candidates},
            active_tol=tol.active,
        )
    )
    return active


def _face_matrix(problem: ParsedProblem, A_basis: np.ndarray, active: list[int]) -> np.ndarray:
    rows = [A_basis]
    if active:
        rows.append(problem.G[active, :])
    return np.vstack(rows) if any(row.shape[0] for row in rows) else np.empty((0, problem.n))


def _multiplier_for_inequalities(
    multipliers: np.ndarray, n_equality_basis: int, active: list[int]
) -> np.ndarray:
    values = (
        np.asarray(multipliers[n_equality_basis : n_equality_basis + len(active)], dtype=float)
        if active
        else np.asarray([])
    )
    if values.size != len(active):
        raise NumericalError(
            "active_set_multiplier_shape_error",
            f"Expected {len(active)} inequality multipliers; got {values.size}.",
        )
    return values


def solve_active_set(problem: ParsedProblem, tol: Tolerances) -> ActiveSetResult:
    events: list[SolverEvent] = []
    phase1 = find_feasible_point(problem, tol)
    events.extend(phase1.events)
    if phase1.x is None or not phase1.feasible:
        return ActiveSetResult(
            x=np.full(problem.n, np.nan),
            status=phase1.status,
            reason=phase1.reason,
            events=events,
            iterations=0,
            active_inequality_indices=[],
            active_inequality_names=[],
            equality_basis_indices=[],
            dependent_equality_indices=[],
            warnings=[] if phase1.status == "infeasible" else [phase1.reason],
        )

    x = phase1.x.copy()
    A_basis, _, equality_basis, dependent_equalities = _preprocess_equalities(problem, tol, events)
    dependent_residual = _check_dependent_equality_consistency(x, problem, dependent_equalities, tol)
    if dependent_residual > tol.primal:
        return ActiveSetResult(
            x=x,
            status="uncertain",
            reason="Rank-deficient equalities are inconsistent at the feasible starting point.",
            events=events
            + [
                _event(
                    0,
                    "preprocess",
                    "dependent_equality_inconsistent",
                    max_residual=dependent_residual,
                    primal_tol=tol.primal,
                )
            ],
            iterations=0,
            active_inequality_indices=[],
            active_inequality_names=[],
            equality_basis_indices=equality_basis.tolist(),
            dependent_equality_indices=dependent_equalities.tolist(),
            warnings=[f"max dependent equality residual={dependent_residual:.3e}"],
        )

    active = _initial_active(x, problem, tol, events)
    max_iterations = problem.max_iterations or max(100, 10 * (problem.n + problem.n_inequalities + problem.n_equalities))
    iteration_count = 0

    for iteration in range(1, max_iterations + 1):
        iteration_count = iteration
        face = _face_matrix(problem, A_basis, active)
        gradient = problem.H @ x + problem.c
        try:
            d, multipliers, stationarity_residual, face_info = solve_face_kkt(
                problem.H, gradient, face, tol.rank, tol.step, float(np.linalg.norm(x))
            )
        except NumericalError as exc:
            events.append(_event(iteration, "phase_2", "numerical_failure", error=exc.message))
            return ActiveSetResult(
                x=x,
                status="failed",
                reason=exc.message,
                events=events,
                iterations=iteration,
                active_inequality_indices=active,
                active_inequality_names=[problem.inequality_names[i] for i in active],
                equality_basis_indices=equality_basis.tolist(),
                dependent_equality_indices=dependent_equalities.tolist(),
            )

        inequality_lambdas = _multiplier_for_inequalities(multipliers, A_basis.shape[0], active)
        events.append(
            _event(
                iteration,
                "phase_2",
                "face_solved",
                active_inequality_indices=active,
                direction=d.tolist(),
                multipliers={
                    problem.inequality_names[active[j]]: float(inequality_lambdas[j])
                    for j in range(len(active))
                },
                stationarity_residual=stationarity_residual,
                **face_info,
            )
        )

        # Positive d along inactive g_i gives g_i(x+alpha d)=g_i x + alpha g_i d.
        inactive = [i for i in range(problem.n_inequalities) if i not in set(active)]
        denominators = problem.G[inactive, :] @ d if inactive else np.asarray([])
        slacks = problem.h[inactive] - problem.G[inactive, :] @ x if inactive else np.asarray([])
        blockers: list[tuple[float, int, float]] = []
        for position, constraint_index in enumerate(inactive):
            denominator = float(denominators[position])
            slack = float(slacks[position])
            if denominator > tol.primal:
                ratio = slack / denominator
                if ratio < -tol.primal:
                    events.append(
                        _event(
                            iteration,
                            "phase_2",
                            "feasibility_lost",
                            constraint_index=constraint_index,
                            slack=slack,
                            denominator=denominator,
                        )
                    )
                    return ActiveSetResult(
                        x=x,
                        status="failed",
                        reason="Line search saw a negative slack before taking a step.",
                        events=events,
                        iterations=iteration,
                        active_inequality_indices=active,
                        active_inequality_names=[problem.inequality_names[i] for i in active],
                        equality_basis_indices=equality_basis.tolist(),
                        dependent_equality_indices=dependent_equalities.tolist(),
                    )
                if ratio <= 1.0 + tol.primal:
                    blockers.append((max(0.0, ratio), constraint_index, denominator))

        # A numerical zero direction is the stationary-face case: remove a
        # negative multiplier when present, otherwise declare KKT. Nonzero
        # directions continue through the line search below.
        zero_direction = face_info.get("zero_direction", False)
        if zero_direction:
            candidates = [
                (j, float(value))
                for j, value in enumerate(inequality_lambdas)
                if value < -tol.dual
            ]
            if candidates:
                local_j, lambda_value = min(candidates, key=lambda item: (item[0],))
                removed = active.pop(local_j)
                events.append(
                    _event(
                        iteration,
                        "phase_2",
                        "negative_multiplier_constraint_removed",
                        constraint_index=removed,
                        constraint_name=problem.inequality_names[removed],
                        multiplier=lambda_value,
                        dual_tol=tol.dual,
                        remaining_active=active,
                    )
                )
                continue

            if stationarity_residual > tol.dual:
                return ActiveSetResult(
                    x=x,
                    status="uncertain",
                    reason="Null-space KKT stationarity residual exceeded tolerance at a stationary face.",
                    events=events,
                    iterations=iteration,
                    active_inequality_indices=active,
                    active_inequality_names=[problem.inequality_names[i] for i in active],
                    equality_basis_indices=equality_basis.tolist(),
                    dependent_equality_indices=dependent_equalities.tolist(),
                    warnings=[f"stationarity_residual={stationarity_residual:.3e}"],
                )

            events.append(_event(iteration, "phase_2", "optimal_face_reached", alpha=1.0))
            break

        if blockers:
            alpha, blocker, denominator = min(blockers, key=lambda item: (item[0], item[1]))
            if is_row_independent(face, problem.G[blocker], tol.rank):
                x = x + alpha * d
                active.append(blocker)
                active.sort()
                events.append(
                    _event(
                        iteration,
                        "phase_2",
                        "blocking_constraint_added",
                        constraint_index=blocker,
                        constraint_name=problem.inequality_names[blocker],
                        alpha=alpha,
                        directional_rate=denominator,
                        active_inequality_indices=active,
                    )
                )
            else:
                x = x + 1.0 * d
                events.append(
                    _event(
                        iteration,
                        "phase_2",
                        "dependent_blocker_skipped",
                        constraint_index=blocker,
                        constraint_name=problem.inequality_names[blocker],
                        alpha=1.0,
                        rank_tol=tol.rank,
                    )
                )
        else:
            alpha = 1.0
            x = x + d
            events.append(_event(iteration, "phase_2", "full_newton_step", alpha=1.0))

        residual_after = violation(x, problem)
        if residual_after > 10.0 * tol.primal:
            return ActiveSetResult(
                x=x,
                status="uncertain",
                reason="Primal feasibility residual grew beyond the allowed numerical tolerance.",
                events=events
                + [
                    _event(
                        iteration,
                        "phase_2",
                        "excessive_primal_residual",
                        max_residual=residual_after,
                        primal_tol=tol.primal,
                    )
                ],
                iterations=iteration,
                active_inequality_indices=active,
                active_inequality_names=[problem.inequality_names[i] for i in active],
                equality_basis_indices=equality_basis.tolist(),
                dependent_equality_indices=dependent_equalities.tolist(),
                warnings=[f"max_primal_violation={residual_after:.3e}"],
            )
    else:
        return ActiveSetResult(
            x=x,
            status="uncertain",
            reason=f"Active-set iteration limit reached after {max_iterations} iterations.",
            events=events,
            iterations=max_iterations,
            active_inequality_indices=active,
            active_inequality_names=[problem.inequality_names[i] for i in active],
            equality_basis_indices=equality_basis.tolist(),
            dependent_equality_indices=dependent_equalities.tolist(),
            warnings=["iteration_limit"],
        )

    final_residual = violation(x, problem)
    warnings: list[str] = []
    if final_residual > tol.primal:
        warnings.append(f"final_primal_violation={final_residual:.3e}")
    if dependent_equalities.size:
        warnings.append(f"{dependent_equalities.size} dependent equalities represented by basis rows")

    return ActiveSetResult(
        x=x,
        status="optimal",
        reason="Active-set solver reached a feasible KKT face.",
        events=events,
        iterations=iteration_count,
        active_inequality_indices=active,
        active_inequality_names=[problem.inequality_names[i] for i in active],
        equality_basis_indices=equality_basis.tolist(),
        dependent_equality_indices=dependent_equalities.tolist(),
        warnings=warnings,
    )
