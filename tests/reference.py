"""Independent reference answers: exhaustive enumeration over small ranges.

Deliberately shares no code with bilp.solver: answers are produced by
itertools + numpy constraint checks, never by the branch-and-bound core,
so cross-checks are meaningful.
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def brute_force_ilp(problem: dict, max_enum: int = 2_000_000) -> dict:
    """Enumerate all integer points; returns status/objective/x."""
    sense = problem.get("sense", "minimize")
    maximize = sense == "maximize"
    c = [float(v) for v in problem["c"]]
    n = len(c)
    integrality = problem.get("integrality") or [1] * n
    if any(v != 1 for v in integrality):
        raise ValueError("reference enumerator only supports pure integer problems")
    bounds = problem.get("bounds") or [[0, None]] * n
    ranges = []
    for i, (lo, hi) in enumerate(bounds):
        if lo is None or hi is None:
            raise ValueError(f"reference enumerator needs finite bounds, check x[{i}]")
        lo_i = math.ceil(lo - 1e-12)
        hi_i = math.floor(hi + 1e-12)
        ranges.append(range(lo_i, hi_i + 1))
    total = 1
    for r in ranges:
        total *= len(r)
    if total > max_enum:
        raise ValueError(f"enumeration space {total} too large for reference check")

    c_arr = np.asarray(c, dtype=float)
    A_ub = np.asarray(problem["A_ub"], dtype=float) if problem.get("A_ub") else None
    b_ub = np.asarray(problem["b_ub"], dtype=float) if problem.get("b_ub") else None
    A_eq = np.asarray(problem["A_eq"], dtype=float) if problem.get("A_eq") else None
    b_eq = np.asarray(problem["b_eq"], dtype=float) if problem.get("b_eq") else None

    tol = 1e-9
    best_obj = None
    best_x = None
    for xs in itertools.product(*ranges):
        x = np.asarray(xs, dtype=float)
        if A_ub is not None and bool(np.any(A_ub @ x > b_ub + tol)):
            continue
        if A_eq is not None and bool(np.any(np.abs(A_eq @ x - b_eq) > tol)):
            continue
        obj = float(c_arr @ x)
        if best_obj is None or (obj > best_obj if maximize else obj < best_obj):
            best_obj = obj
            best_x = [int(v) for v in xs]
    return {
        "status": "optimal" if best_x is not None else "infeasible",
        "objective": best_obj,
        "x": best_x,
    }
