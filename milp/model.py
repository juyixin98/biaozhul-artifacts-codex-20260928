"""Parse and validate the external LP/MILP representation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from .errors import InputError
from .schemas import Direction, LPProblem, Sense, VarType


def _finite(value: float, field: str) -> float:
    if not math.isfinite(float(value)):
        raise InputError(f"{field} must be finite")
    return float(value)


@dataclass(frozen=True)
class ParsedProblem:
    name: str
    sense: str
    names: list[str]
    integer: np.ndarray
    original_lower: np.ndarray
    original_upper: np.ndarray
    internal_lower: np.ndarray
    internal_upper: np.ndarray
    objective: np.ndarray
    a_ub: csr_matrix
    b_ub: np.ndarray
    a_eq: csr_matrix
    b_eq: np.ndarray
    constraint_names: list[str]

    @property
    def n(self) -> int:
        return len(self.names)

    def variable_bounds(self, overrides: dict[int, tuple[float, float]] | None = None) -> tuple[np.ndarray, np.ndarray]:
        lower = self.internal_lower.copy()
        upper = self.internal_upper.copy()
        for index, (new_lower, new_upper) in (overrides or {}).items():
            lower[index] = max(lower[index], float(new_lower))
            upper[index] = min(upper[index], float(new_upper))
        return lower, upper


def parse_problem(problem: LPProblem) -> ParsedProblem:
    names = [var.name for var in problem.variables]
    if len(names) != len(set(names)):
        raise InputError("variable names must be unique")
    index = {name: i for i, name in enumerate(names)}

    unknown = set(problem.objective) - set(names)
    if unknown:
        raise InputError(f"objective contains unknown variables: {sorted(unknown)}")
    for coefficient in problem.objective.values():
        _finite(coefficient, "objective coefficient")

    original_objective = np.zeros(len(names), dtype=float)
    for name, coefficient in problem.objective.items():
        original_objective[index[name]] = float(coefficient)
    internal_objective = original_objective.copy()
    if problem.sense == Direction.maximize:
        internal_objective = -internal_objective

    original_lower = np.full(len(names), -np.inf, dtype=float)
    original_upper = np.full(len(names), np.inf, dtype=float)

    for i, var in enumerate(problem.variables):
        lower = 0.0 if var.type in {VarType.integer, VarType.binary} else -np.inf
        upper = 1.0 if var.type is VarType.binary else np.inf
        if var.lower is not None:
            lower = _finite(var.lower, f"lower bound for {var.name}")
        if var.upper is not None:
            upper = _finite(var.upper, f"upper bound for {var.name}")
        if lower > upper:
            raise InputError(f"variable {var.name} has lower bound greater than upper bound")
        original_lower[i] = lower
        original_upper[i] = upper

    integer_flags = np.array([var.type in {VarType.integer, VarType.binary} for var in problem.variables], dtype=bool)
    internal_lower = original_lower.copy()
    internal_upper = original_upper.copy()

    for i, name in enumerate(names):
        if integer_flags[i]:
            if not np.isfinite(original_lower[i]) or not np.isfinite(original_upper[i]):
                raise InputError(f"integer variable {name} must declare finite lower and upper bounds")
            internal_lower[i] = np.ceil(original_lower[i] - 1e-12)
            internal_upper[i] = np.floor(original_upper[i] + 1e-12)
            if internal_lower[i] > internal_upper[i]:
                # The rounded integer domain is empty; the solver detects it as model infeasibility.
                continue

    ub_rows: list[list[tuple[int, float]]] = []
    ub_rhs: list[float] = []
    eq_rows: list[list[tuple[int, float]]] = []
    eq_rhs: list[float] = []
    constraint_names: list[str] = []

    for position, constraint in enumerate(problem.constraints):
        cname = constraint.name or f"constraint_{position}"
        if cname in constraint_names:
            raise InputError(f"duplicate constraint name: {cname}")
        constraint_names.append(cname)

        unknown = set(constraint.terms) - set(names)
        if unknown:
            raise InputError(f"constraint {cname} contains unknown variables: {sorted(unknown)}")

        row = [(index[name], _finite(value, f"coefficient in {cname}")) for name, value in constraint.terms.items()]
        rhs = _finite(constraint.rhs, f"rhs in {cname}")
        if constraint.sense is Sense.le:
            ub_rows.append(row)
            ub_rhs.append(rhs)
        elif constraint.sense is Sense.ge:
            ub_rows.append([(j, -value) for j, value in row])
            ub_rhs.append(-rhs)
        else:
            eq_rows.append(row)
            eq_rhs.append(rhs)

    def sparse(rows: list[list[tuple[int, float]]]) -> csr_matrix:
        data: list[float] = []
        indices: list[int] = []
        indptr = [0]
        for row in rows:
            for j, value in row:
                indices.append(j)
                data.append(value)
            indptr.append(len(data))
        return csr_matrix((np.asarray(data, dtype=float), np.asarray(indices), np.asarray(indptr)), shape=(len(rows), len(names)))

    return ParsedProblem(
        name=problem.name,
        sense=problem.sense.value,
        names=names,
        integer=integer_flags,
        original_lower=original_lower,
        original_upper=original_upper,
        internal_lower=internal_lower,
        internal_upper=internal_upper,
        objective=internal_objective,
        a_ub=sparse(ub_rows),
        b_ub=np.asarray(ub_rhs, dtype=float),
        a_eq=sparse(eq_rows),
        b_eq=np.asarray(eq_rhs, dtype=float),
        constraint_names=constraint_names,
    )
