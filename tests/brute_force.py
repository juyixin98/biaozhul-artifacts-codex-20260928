"""Independent exhaustive oracle for small bounded integer models.

This oracle intentionally does not import or call the branch-and-bound solver.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from milp.schemas import LPProblem, Sense, VarType


def enumerate_integer_solutions(problem: LPProblem, *, tolerance: float = 1e-9) -> list[dict[str, float]]:
    domains: list[list[int]] = []
    names: list[str] = []
    for variable in problem.variables:
        if variable.type is VarType.continuous:
            raise ValueError("oracle supports small integer/binary models only")
        names.append(variable.name)
        lower = 0 if variable.lower is None else int(np.ceil(variable.lower))
        upper = 1 if variable.type is VarType.binary else int(np.floor(variable.upper))
        if variable.type is VarType.integer:
            lower = int(np.ceil(variable.lower if variable.lower is not None else 0.0))
            upper = int(np.floor(variable.upper if variable.upper is not None else 0.0))
        if lower > upper:
            return []
        domains.append(list(range(lower, upper + 1)))

    feasible: list[dict[str, float]] = []
    for assignment in product(*domains):
        values = dict(zip(names, assignment, strict=True))
        ok = True
        for constraint in problem.constraints:
            lhs = sum(values[name] * coefficient for name, coefficient in constraint.terms.items())
            rhs = constraint.rhs
            if constraint.sense is Sense.le:
                ok = lhs <= rhs + tolerance
            elif constraint.sense is Sense.ge:
                ok = lhs >= rhs - tolerance
            else:
                ok = abs(lhs - rhs) <= tolerance
            if not ok:
                break
        if ok:
            feasible.append({name: float(value) for name, value in values.items()})
    return feasible


def oracle(problem: LPProblem) -> tuple[float | None, list[dict[str, float]]]:
    feasible = enumerate_integer_solutions(problem)
    if not feasible:
        return None, []
    objectives = [sum(point[name] * coefficient for name, coefficient in problem.objective.items()) for point in feasible]
    best = max(objectives) if problem.sense.value == "maximize" else min(objectives)
    return float(best), [point for point, value in zip(feasible, objectives, strict=True) if abs(value - best) <= 1e-9]


def knapsack() -> LPProblem:
    return LPProblem(
        name="knapsack",
        sense="maximize",
        objective={"x1": 5, "x2": 3, "x3": 2},
        variables=[
            {"name": "x1", "type": "binary"},
            {"name": "x2", "type": "binary"},
            {"name": "x3", "type": "binary"},
        ],
        constraints=[
            {"name": "weight", "terms": {"x1": 4, "x2": 3, "x3": 2}, "sense": "<=", "rhs": 5},
        ],
    )
