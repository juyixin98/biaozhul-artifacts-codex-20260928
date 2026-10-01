from dataclasses import dataclass
import math

import numpy as np

from app.config import Tolerances
from app.errors import InvalidProblemError
from app.models import LinearConstraint, SolveRequest, FeasibilityRequest


@dataclass(frozen=True)
class ParsedProblem:
    n: int
    H: np.ndarray
    c: np.ndarray
    G: np.ndarray
    h: np.ndarray
    A: np.ndarray
    b: np.ndarray
    variable_names: list[str]
    inequality_names: list[str]
    equality_names: list[str]
    source_descriptions: list[str]
    initial_point: np.ndarray | None
    max_iterations: int | None

    @property
    def n_inequalities(self) -> int:
        return len(self.h)

    @property
    def n_equalities(self) -> int:
        return len(self.b)


def _finite_vector(values: np.ndarray, what: str) -> None:
    if not np.all(np.isfinite(values)):
        raise InvalidProblemError("non_finite_value", f"{what} contains NaN or infinite values.")


def _constraint_arrays(
    constraints: list[LinearConstraint], n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    G_rows: list[np.ndarray] = []
    h_values: list[float] = []
    A_rows: list[np.ndarray] = []
    b_values: list[float] = []
    ineq_names: list[str] = []
    eq_names: list[str] = []
    descriptions: list[str] = []
    seen_names: set[str] = set()

    for index, constraint in enumerate(constraints):
        if len(constraint.coefficients) != n:
            raise InvalidProblemError(
                "constraint_dimension_mismatch",
                f"Constraint {index} has {len(constraint.coefficients)} coefficients; expected {n}.",
            )
        row = np.asarray(constraint.coefficients, dtype=float)
        _finite_vector(row, f"Constraint {index}")
        if not math.isfinite(constraint.rhs):
            raise InvalidProblemError("non_finite_value", f"Constraint {index} rhs is not finite.")

        supplied_name = constraint.name or f"constraint_{index}"
        if supplied_name in seen_names:
            raise InvalidProblemError(
                "duplicate_constraint_name", f"Duplicate constraint name: {supplied_name}."
            )
        seen_names.add(supplied_name)

        if constraint.op in {"=", "=="}:
            A_rows.append(row)
            b_values.append(float(constraint.rhs))
            eq_names.append(supplied_name)
            descriptions.append(f"{supplied_name}: {_row_text(row)} = {constraint.rhs:g}")
        elif constraint.op == "<=":
            G_rows.append(row)
            h_values.append(float(constraint.rhs))
            ineq_names.append(supplied_name)
            descriptions.append(f"{supplied_name}: {_row_text(row)} <= {constraint.rhs:g}")
        else:  # ">="
            G_rows.append(-row)
            h_values.append(-float(constraint.rhs))
            ineq_names.append(supplied_name)
            descriptions.append(f"{supplied_name}: {_row_text(row)} >= {constraint.rhs:g}")

    G = np.vstack(G_rows) if G_rows else np.empty((0, n), dtype=float)
    h = np.asarray(h_values, dtype=float)
    A = np.vstack(A_rows) if A_rows else np.empty((0, n), dtype=float)
    b = np.asarray(b_values, dtype=float)
    return G, h, A, b, ineq_names, eq_names, descriptions


def _row_text(row: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:g}" for value in row) + "]"


def _variable_names(names: list[str] | None, n: int) -> list[str]:
    if names is None:
        return [f"x{i}" for i in range(n)]
    if len(names) != n:
        raise InvalidProblemError(
            "variable_name_dimension_mismatch",
            f"{len(names)} variable names supplied; expected {n}.",
        )
    if len(set(names)) != n:
        raise InvalidProblemError("duplicate_variable_name", "Variable names must be unique.")
    return names


def _check_hessian(H: np.ndarray, tol: Tolerances) -> np.ndarray:
    _finite_vector(H, "Hessian")
    asym_scale = max(1.0, float(np.linalg.norm(H)))
    asymmetry = np.linalg.norm(H - H.T, ord=np.inf)
    if asymmetry > tol.symmetry * asym_scale:
        raise InvalidProblemError(
            "hessian_not_symmetric",
            f"Hessian symmetry residual {asymmetry:.3e} exceeds tolerance {tol.symmetry:.3e}.",
        )
    Hs = 0.5 * (H + H.T)
    eigvals = np.linalg.eigvalsh(Hs)
    minimum = float(eigvals[0])
    scale = max(1.0, float(np.max(np.abs(eigvals))))
    if minimum <= tol.positive_definite * scale:
        raise InvalidProblemError(
            "hessian_not_positive_definite",
            f"Smallest Hessian eigenvalue is {minimum:.3e}; it must be positive.",
        )
    return Hs


def parse_qp_request(request: SolveRequest, tol: Tolerances) -> ParsedProblem:
    H_input = np.asarray(request.objective.quadratic, dtype=float)
    if H_input.ndim != 2 or H_input.shape[0] != H_input.shape[1]:
        raise InvalidProblemError(
            "hessian_shape_invalid", "objective.quadratic must be a square matrix."
        )
    n = H_input.shape[0]
    if n < 1:
        raise InvalidProblemError("dimension_missing", "The problem must have at least one variable.")
    H = _check_hessian(H_input, tol)

    if request.objective.linear is None:
        c = np.zeros(n, dtype=float)
    else:
        if len(request.objective.linear) != n:
            raise InvalidProblemError(
                "linear_dimension_mismatch",
                f"objective.linear has length {len(request.objective.linear)}; expected {n}.",
            )
        c = np.asarray(request.objective.linear, dtype=float)
        _finite_vector(c, "Linear objective vector")

    G, h, A, b, ineq_names, eq_names, descriptions = _constraint_arrays(request.constraints, n)
    variable_names = _variable_names(request.variable_names, n)

    x0 = None
    if request.initial_point is not None:
        if len(request.initial_point) != n:
            raise InvalidProblemError(
                "initial_point_dimension_mismatch",
                f"initial_point has length {len(request.initial_point)}; expected {n}.",
            )
        x0 = np.asarray(request.initial_point, dtype=float)
        _finite_vector(x0, "initial_point")

    return ParsedProblem(
        n=n,
        H=H,
        c=c,
        G=G,
        h=h,
        A=A,
        b=b,
        variable_names=variable_names,
        inequality_names=ineq_names,
        equality_names=eq_names,
        source_descriptions=descriptions,
        initial_point=x0,
        max_iterations=request.max_iterations,
    )


def parse_feasibility_request(
    request: FeasibilityRequest, tol: Tolerances
) -> tuple[int, ParsedProblem | None]:
    inferred_n = request.n
    if request.constraints and inferred_n is None:
        inferred_n = len(request.constraints[0].coefficients)
    if inferred_n is None and request.initial_point is not None:
        inferred_n = len(request.initial_point)
    if inferred_n is None:
        raise InvalidProblemError("dimension_missing", "Provide n, a constraint, or initial_point.")

    if request.constraints:
        lengths = {len(row.coefficients) for row in request.constraints}
        if lengths != {inferred_n}:
            raise InvalidProblemError(
                "constraint_dimension_mismatch",
                "Every feasibility constraint must have exactly n coefficients.",
            )

    H = np.eye(inferred_n, dtype=float)
    c = np.zeros(inferred_n, dtype=float)
    G, h, A, b, ineq_names, eq_names, descriptions = _constraint_arrays(
        request.constraints, inferred_n
    )
    variable_names = _variable_names(request.variable_names, inferred_n)

    x0 = None
    if request.initial_point is not None:
        if len(request.initial_point) != inferred_n:
            raise InvalidProblemError(
                "initial_point_dimension_mismatch",
                f"initial_point has length {len(request.initial_point)}; expected {inferred_n}.",
            )
        x0 = np.asarray(request.initial_point, dtype=float)
        _finite_vector(x0, "initial_point")

    return inferred_n, ParsedProblem(
        n=inferred_n,
        H=H,
        c=c,
        G=G,
        h=h,
        A=A,
        b=b,
        variable_names=variable_names,
        inequality_names=ineq_names,
        equality_names=eq_names,
        source_descriptions=descriptions,
        initial_point=x0,
        max_iterations=None,
    )
