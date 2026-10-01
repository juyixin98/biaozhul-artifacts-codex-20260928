"""Convert a user-level LP into nonnegative equality standard form."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np

from .models import Operator, SolveRequest


class VarKind(str, Enum):
    NORMAL = "normal"
    UPPER_ONLY = "upper_only"
    FREE_PLUS = "free_plus"
    FREE_MINUS = "free_minus"


@dataclass(frozen=True)
class VariableRep:
    original: int
    kind: VarKind
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class RowSource:
    original: int | None
    kind: str
    op: Operator | None = None
    sign: float = 1.0


@dataclass(frozen=True)
class ColumnSource:
    kind: str
    row: int | None = None
    variable: int | None = None


@dataclass
class StandardLP:
    n_original: int
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    original_c: np.ndarray
    constant: float
    sense: str
    basis: list[int]
    variable_reps: list[VariableRep]
    rows: list[RowSource]
    columns: list[ColumnSource]
    original_names: list[str]
    constraint_names: list[str]
    original_bounds: list[tuple[float | None, float | None]]
    original_ops: list[Operator]
    original_rhs: np.ndarray
    original_A: np.ndarray
    slack_cols: dict[int, int] = field(default_factory=dict)
    artificial_cols: dict[int, int] = field(default_factory=dict)

    def original_objective_vector(self) -> np.ndarray:
        return self.original_c.copy()


def _bounded(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        raise ValueError("bounds must not be NaN")
    if math.isinf(number):
        return None
    return number


def standardize(request: SolveRequest) -> StandardLP:
    n = len(request.objective)
    original_names = request.variable_names or [f"x{i}" for i in range(n)]
    raw_bounds = request.bounds or [(0.0, None)] * n
    bounds: list[tuple[float | None, float | None]] = []
    for pair in raw_bounds:
        lb, ub = pair if pair is not None else (None, None)
        lb, ub = _bounded(lb), _bounded(ub)
        if lb is not None and ub is not None and lb > ub + 1e-12:
            raise ValueError("lower bound is greater than upper bound")
        bounds.append((lb, ub))

    m = len(request.constraints)
    original_A = np.array(
        [constraint.coefficients for constraint in request.constraints],
        dtype=float,
        ndmin=2,
    )
    if m == 0:
        original_A = np.empty((0, n), dtype=float)
    original_rhs = np.array([constraint.rhs for constraint in request.constraints], dtype=float)
    original_ops: list[Operator] = [constraint.op for constraint in request.constraints]
    constraint_names = [
        constraint.name or f"constraint_{i}"
        for i, constraint in enumerate(request.constraints)
    ]

    objective = np.array(request.objective, dtype=float)
    if request.sense == "max":
        objective = -objective
    constant = 0.0

    reps: list[VariableRep] = []
    structural_columns: list[list[int]] = [[] for _ in range(n)]
    entries: list[dict[int, float]] = []
    row_sources: list[RowSource] = [
        RowSource(i, "constraint", original_ops[i], 1.0) for i in range(m)
    ]
    shifted_rhs = original_rhs.astype(float).copy()

    def add_structural(
        rep: VariableRep, column_entries: dict[int, float], cost: float
    ) -> int:
        column = len(reps)
        reps.append(rep)
        structural_columns[rep.original].append(column)
        while len(entries) <= column:
            entries.append({})
        for row, value in column_entries.items():
            if value != 0.0:
                entries[column][row] = float(value)
        costs.append(float(cost))
        return column

    costs: list[float] = []
    for j, (lb, ub) in enumerate(bounds):
        original_entries = {
            row: float(original_A[row, j])
            for row in range(m)
            if original_A[row, j] != 0.0
        }
        if lb is None and ub is None:
            add_structural(VariableRep(j, VarKind.FREE_PLUS, None, None), original_entries, objective[j])
            add_structural(
                VariableRep(j, VarKind.FREE_MINUS, None, None),
                {row: -value for row, value in original_entries.items()},
                -objective[j],
            )
        elif lb is None:
            constant += float(objective[j]) * float(ub)
            column = add_structural(
                VariableRep(j, VarKind.UPPER_ONLY, None, float(ub)),
                {row: -value for row, value in original_entries.items()},
                -objective[j],
            )
            # x = ub - y, y >= 0.  If ub < 0 this also has a finite lower
            # bound, represented as y = -ub + z, z >= 0.
            if float(ub) < 0.0:
                shift = -float(ub)
                constant -= 2.0 * float(objective[j]) * shift
                for row in range(m):
                    shifted_rhs[row] += 2.0 * float(original_A[row, j]) * shift
                reps[column] = VariableRep(j, VarKind.UPPER_ONLY, None, float(ub))
        else:
            constant += float(objective[j]) * float(lb)
            for row in range(m):
                shifted_rhs[row] -= float(original_A[row, j]) * float(lb)
            width = None if ub is None else float(ub) - float(lb)
            column = add_structural(
                VariableRep(j, VarKind.NORMAL, 0.0, width), original_entries, objective[j]
            )
            if width is not None:
                row = len(row_sources)
                row_sources.append(RowSource(None, "upper_bound", "<=", 1.0))
                entries[column][row] = 1.0

    n_structural = len(reps)
    rhs = list(shifted_rhs) + [0.0 for _ in range(len(row_sources) - m)]
    for column in range(n_structural):
        rep = reps[column]
        if rep.kind == VarKind.NORMAL and rep.upper is not None:
            bound_rows = [
                row
                for row, source in enumerate(row_sources)
                if source.kind == "upper_bound" and entries[column].get(row) == 1.0
            ]
            if bound_rows:
                rhs[bound_rows[0]] = float(rep.upper)

    slack_cols: dict[int, int] = {}
    columns: list[ColumnSource] = [
        ColumnSource("structural", variable=rep.original) for rep in reps
    ]

    def add_auxiliary(row: int, value: float, kind: str, cost: float = 0.0) -> int:
        column = len(reps)
        reps.append(VariableRep(-1, VarKind.NORMAL, 0.0, None))
        entries.append({row: value})
        costs.append(cost)
        columns.append(ColumnSource(kind, row=row))
        return column

    # Convert inequalities to equalities with nonnegative slack/surplus.
    # A >= b is first multiplied by -1, yielding -A y + s = -b.
    for row in range(m):
        op = row_sources[row].op
        if op == "<=":
            continue
        elif op == ">=":
            for column in range(n_structural):
                col_entries = entries[column]
                if row in col_entries:
                    col_entries[row] = -col_entries[row]
            rhs[row] = -rhs[row]
            row_sources[row] = RowSource(row, "constraint", ">=", -1.0)
            continue

    # Sign-normalize RHS so artificial variables initialize at max(0, |b|).
    for row in range(len(row_sources)):
        if rhs[row] < 0.0:
            rhs[row] = -rhs[row]
            for column in range(n_structural):
                col_entries = entries[column]
                if row in col_entries:
                    col_entries[row] = -col_entries[row]
            source = row_sources[row]
            sign = -source.sign
            row_sources[row] = RowSource(source.original, source.kind, source.op, sign)

    # Add slack/surplus columns after row sign normalization.  Their sign is
    # selected for the final canonical inequality, not the original orientation.
    for row, source in enumerate(row_sources):
        if source.kind == "constraint":
            if source.op == "=":
                continue
            slack_value = source.sign if source.op == "<=" else -source.sign
        else:
            slack_value = 1.0
        col = add_auxiliary(row, slack_value, "slack")
        slack_cols[row] = col

    artificial_cols: dict[int, int] = {}
    basis: list[int] = []
    for row in range(len(row_sources)):
        col = add_auxiliary(row, 1.0, "artificial")
        artificial_cols[row] = col
        basis.append(col)

    A = np.zeros((len(row_sources), len(reps)), dtype=float)
    for column, col_entries in enumerate(entries):
        for row, value in col_entries.items():
            A[row, column] = value

    return StandardLP(
        n_original=n,
        A=A,
        b=np.array(rhs, dtype=float),
        c=np.array(costs, dtype=float),
        original_c=np.array(request.objective, dtype=float),
        constant=float(constant),
        sense=request.sense,
        basis=basis,
        variable_reps=reps,
        rows=row_sources,
        columns=columns,
        original_names=original_names,
        constraint_names=constraint_names,
        original_bounds=bounds,
        original_ops=original_ops,
        original_rhs=original_rhs,
        original_A=original_A,
        slack_cols=slack_cols,
        artificial_cols=artificial_cols,
    )
