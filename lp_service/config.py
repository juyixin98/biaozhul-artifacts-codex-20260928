"""Runtime configuration and numerical tolerances."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverConfig:
    """Tolerances used by the simplex kernel and evidence checks."""

    pivot_tol: float = 1e-12
    zero_tol: float = 1e-9
    uncertain_tol: float = 1e-7
    evidence_tol: float = 1e-7
    max_iterations: int = 10_000

    def __post_init__(self) -> None:
        if self.pivot_tol <= 0 or self.zero_tol <= 0 or self.uncertain_tol <= 0:
            raise ValueError("tolerances must be positive")
        if self.pivot_tol > self.zero_tol or self.zero_tol > self.uncertain_tol:
            raise ValueError("require pivot_tol <= zero_tol <= uncertain_tol")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
