"""Internal numeric domain types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

Vector = np.ndarray
Matrix = np.ndarray


@dataclass(frozen=True)
class QuadraticWorker:
    worker_id: str
    P: Matrix
    q: Vector
    r: float
    sample_count: int


@dataclass(frozen=True)
class QuadraticModel:
    dimension: int
    workers: tuple[QuadraticWorker, ...]

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(w.worker_id for w in self.workers)

    def worker(self, worker_id: str) -> QuadraticWorker:
        for item in self.workers:
            if item.worker_id == worker_id:
                return item
        raise KeyError(worker_id)


@dataclass(frozen=True)
class ADMMConfig:
    initial_rho: float
    eps_abs: float
    eps_rel: float
    max_iterations: int
    deadline_ms: int
    rho_schedule: Mapping[int, float]


@dataclass(frozen=True)
class LocalUpdate:
    worker_id: str
    iteration: int
    rho: float
    z_pre: Vector
    u_pre: Vector
    x: Vector
    objective: float
    stationarity_residual: float
    solve_seconds: float


@dataclass(frozen=True)
class RoundUpdate:
    iteration: int
    rho: float
    rho_previous: float
    z_pre: Vector
    z: Vector
    x: dict[str, Vector]
    u_pre: dict[str, Vector]
    u: dict[str, Vector]
    primal_residual: float
    dual_residual: float
    eps_primal: float
    eps_dual: float
    objectives: dict[str, float]
    objective_total: float


@dataclass(frozen=True)
class ConvergenceDecision:
    converged: bool
    primal_ok: bool
    dual_ok: bool
    reason: str
