"""Internal data contracts for parsed problems and solver state."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

import numpy as np

from .errors import InputError

RunStatus = Literal["pending", "running", "converged", "max_iterations", "failed"]


@dataclass(frozen=True)
class QuadraticModel:
    """f_i(x_i)=0.5*x_i' Q_i x_i + q_i' x_i, with positive definite Q_i."""

    Q: np.ndarray
    q: np.ndarray

    def objective(self, x: np.ndarray) -> float:
        return float(0.5 * x @ self.Q @ x + self.q @ x)


@dataclass(frozen=True)
class Problem:
    models: tuple[QuadraticModel, QuadraticModel]
    dimension: int

    @property
    def objective_constant(self) -> float:
        return 0.0

    def total_objective(self, x: np.ndarray) -> float:
        return sum(model.objective(x) for model in self.models) + self.objective_constant


@dataclass(frozen=True)
class AdaptiveRho:
    """Balanced residual schedule.

    Rho is changed before the local update at the beginning of rounds in
    ``change_rounds``. The dual is immediately rescaled using old/new rho.
    """

    mu: float = 10.0
    tau_incr: float = 2.0
    tau_decr: float = 2.0
    change_rounds: frozenset[int] = field(default_factory=frozenset)

    @classmethod
    def from_api(cls, data: dict[str, Any] | None) -> "AdaptiveRho | None":
        if data is None:
            return None
        try:
            change_rounds = frozenset(int(v) for v in data.get("change_rounds", []))
            return cls(
                mu=float(data.get("mu", 10.0)),
                tau_incr=float(data.get("tau_incr", 2.0)),
                tau_decr=float(data.get("tau_decr", 2.0)),
                change_rounds=change_rounds,
            )
        except (TypeError, ValueError) as exc:
            raise InputError("Adaptive rho options are not numeric.", {"field": "adaptive_rho"}) from exc

    def validate(self) -> None:
        if self.mu <= 1.0:
            raise InputError("adaptive_rho.mu must be > 1.")
        if self.tau_incr <= 1.0 or self.tau_decr <= 1.0:
                raise InputError("adaptive rho factors must be > 1.")
        if any(round_number < 1 for round_number in self.change_rounds):
            raise InputError("adaptive rho change rounds must use 1-based round numbers.")

    def next_rho(self, *, round_number: int, primal_residual: float, dual_residual: float, rho: float) -> float:
        if round_number not in self.change_rounds:
            return rho
        if primal_residual > self.mu * dual_residual:
            return rho * self.tau_incr
        if dual_residual > self.mu * primal_residual:
            return rho / self.tau_decr
        return rho


@dataclass(frozen=True)
class Tolerances:
    primal_abs: float
    dual_abs: float
    primal_rel: float = 0.0
    dual_rel: float = 0.0

    def primal_threshold(self, x_norm: float, z_norm: float) -> float:
        return self.primal_abs + self.primal_rel * max(x_norm, z_norm)

    def dual_threshold(self, dual_norm: float) -> float:
        return self.dual_abs + self.dual_rel * dual_norm


@dataclass(frozen=True)
class SolverConfig:
    max_iterations: int
    initial_rho: float
    tol: Tolerances
    adaptive_rho: AdaptiveRho | None = None


@dataclass
class SolverState:
    run_id: str
    problem: Problem
    config: SolverConfig
    status: RunStatus
    round_number: int
    rho: float
    x: list[np.ndarray]
    z: np.ndarray
    u: list[np.ndarray]
    pending: dict[int, np.ndarray]
    trajectory: list[dict[str, Any]]
    last_timeout: dict[str, Any] | None = None
    failure_reason: str | None = None

    def public_rho_schedule(self) -> float:
        """Rho that the next round will use after any scheduled adjustment."""
        adaptive = self.config.adaptive_rho
        if adaptive is None or not self.trajectory:
            return self.rho
        latest = self.trajectory[-1]
        return adaptive.next_rho(
            round_number=self.round_number,
            primal_residual=float(latest["primal_residual"]),
            dual_residual=float(latest["dual_residual"]),
            rho=self.rho,
        )


def positive_finite(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise InputError(f"{name} must be a positive finite number.")
    return float(value)
