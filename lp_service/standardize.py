"""Convert a general LP into non-negative variables plus equalities.

Column and row metadata make the transformation auditable. The mapping layer
uses those records to recover original variables, duals and Farkas evidence.
"""

from dataclasses import dataclass

import numpy as np

from .parser import LPProblem


@dataclass(frozen=True)
class ColumnMeta:
    kind: str  # shifted, upper_only, positive_part, negative_part, slack, artificial
    variable: int
    slack_sign: float = 0.0


@dataclass
class RowMeta:
    kind: str  # constraint or upper
    original_index: int
    row_index: int
    # +1 if the row kept its direction, -1 if multiplied through by -1.
    normalization_sign: float
    # Optimal equality dual follows normalization sign; the Farkas
    # multiplier for an inequality gets the opposite sign when rows flip.
    dual_factor: float
    cert_factor: float
    slack_sign: float = 0.0
    initial_basic: int = -1
    initial_artificial: bool = False


@dataclass
class StandardModel:
    c: np.ndarray
    d: np.ndarray
    A: np.ndarray
    columns: list[ColumnMeta]
    rows: list[RowMeta]
    lower: np.ndarray
    upper: np.ndarray
    objective_constant: float
    sense: str
    original_n: int
    core_column_count: int


def standardize(problem: LPProblem) -> StandardModel:
    n = problem.c.size
    columns: list[ColumnMeta] = []
    T_columns: list[np.ndarray] = []
    c_core: list[float] = []
    shifts = np.zeros(n, dtype=float)
    double_bounded = np.zeros(n, dtype=bool)
    internal_c = -problem.c if problem.sense == "max" else problem.c.copy()
    objective_constant = 0.0

    for j in range(n):
        lb, ub = float(problem.lower[j]), float(problem.upper[j])
        if lb == ub:
            shifts[j] = lb
            objective_constant += float(internal_c[j] * lb)
        elif lb > -np.inf and ub < np.inf:
            double_bounded[j] = True
            shifts[j] = lb
            _identity_column(T_columns, n, j)
            c_core.append(float(internal_c[j]))
            objective_constant += float(internal_c[j] * lb)
            columns.append(ColumnMeta("shifted", j))
        elif lb > -np.inf:
            shifts[j] = lb
            _identity_column(T_columns, n, j)
            c_core.append(float(internal_c[j]))
            objective_constant += float(internal_c[j] * lb)
            columns.append(ColumnMeta("shifted", j))
        elif ub < np.inf:
            shifts[j] = ub
            vec = np.zeros(n, dtype=float)
            vec[j] = -1.0
            T_columns.append(vec)
            c_core.append(float(-internal_c[j]))
            objective_constant += float(internal_c[j] * ub)
            columns.append(ColumnMeta("upper_only", j))
        else:
            pos = np.zeros(n, dtype=float)
            neg = np.zeros(n, dtype=float)
            pos[j], neg[j] = 1.0, -1.0
            T_columns.extend([pos, neg])
            c_core.extend([float(internal_c[j]), float(-internal_c[j])])
            columns.append(ColumnMeta("positive_part", j))
            columns.append(ColumnMeta("negative_part", j))

    transform = np.vstack(T_columns).T if T_columns else np.empty((n, 0), dtype=float)
    transformed_rows: list[np.ndarray] = []
    transformed_d: list[float] = []
    rows: list[RowMeta] = []

    for i in range(problem.A.shape[0]):
        original_row = problem.A[i]
        row = original_row @ transform
        rhs = float(problem.b[i] - original_row @ shifts)
        sense = str(problem.constraint_senses[i])
        sign = 1.0
        if rhs < 0.0:
            row = -row
            rhs = -rhs
            sign = -1.0

        slack_sign = 0.0
        if sense == "le":
            slack_sign = sign
        elif sense == "ge":
            slack_sign = -sign
        if sense == "eq":
            cert_factor = sign
        else:
            # Tableau multipliers are for nonneg slack combinations. Convert
            # them to the conventional nonnegative >= Farkas multiplier.
            cert_factor = -sign if sense == "ge" else sign

        transformed_rows.append(row)
        transformed_d.append(rhs)
        rows.append(
            RowMeta(
                kind="constraint",
                original_index=i,
                row_index=len(rows),
                normalization_sign=sign,
                dual_factor=sign,
                cert_factor=cert_factor,
                slack_sign=slack_sign,
            )
        )

    # A shifted variable with both bounds needs an explicit upper row. An
    # upper-only variable needs none: x = u - z, z >= 0 already means x <= u.
    for j in np.flatnonzero(double_bounded):
        lb, ub = float(problem.lower[j]), float(problem.upper[j])
        row = np.zeros(transform.shape[1], dtype=float)
        row[_column_index_for_variable(columns, int(j))] = 1.0
        transformed_rows.append(row)
        transformed_d.append(ub - lb)
        rows.append(
            RowMeta(
                kind="upper",
                original_index=int(j),
                row_index=len(rows),
                normalization_sign=1.0,
                dual_factor=1.0,
                cert_factor=1.0,
                slack_sign=1.0,
            )
        )

    core_A = np.vstack(transformed_rows) if transformed_rows else np.empty((0, transform.shape[1]))
    d = np.asarray(transformed_d, dtype=float)
    m = len(rows)
    core_count = core_A.shape[1]

    slack_A = np.zeros((m, m), dtype=float)
    artificial_A = np.zeros((m, m), dtype=float)
    for r, meta in enumerate(rows):
        if meta.slack_sign:
            slack_A[r, r] = meta.slack_sign
            meta.initial_basic = core_count + r
            columns.append(ColumnMeta("slack", -1, meta.slack_sign))
        else:
            artificial_A[r, r] = 1.0
            meta.initial_artificial = True
            meta.initial_basic = core_count + m + r

    for _ in range(sum(meta.initial_artificial for meta in rows)):
        columns.append(ColumnMeta("artificial", -1))

    blocks = [core_A]
    if m:
        blocks.extend([slack_A, artificial_A])
    A = np.hstack(blocks)
    c = np.concatenate([np.asarray(c_core, dtype=float), np.zeros(2 * m, dtype=float)])

    return StandardModel(
        c=c,
        d=d,
        A=A,
        columns=columns,
        rows=rows,
        lower=problem.lower.copy(),
        upper=problem.upper.copy(),
        objective_constant=objective_constant,
        sense=problem.sense,
        original_n=n,
        core_column_count=core_count,
    )


def _identity_column(columns: list[np.ndarray], n: int, j: int) -> None:
    vec = np.zeros(n, dtype=float)
    vec[j] = 1.0
    columns.append(vec)


def _column_index_for_variable(columns: list[ColumnMeta], variable: int) -> int:
    for i, meta in enumerate(columns):
        if meta.variable == variable and meta.kind == "shifted":
            return i
    raise RuntimeError(f"missing shifted column for variable {variable}")
