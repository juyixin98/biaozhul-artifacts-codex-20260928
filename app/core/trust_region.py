"""Bound-aware truncated conjugate-gradient trust-region subproblem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrustRegionStep:
    step: np.ndarray
    predicted_reduction: float
    cg_iterations: int
    hit_trust_boundary: bool
    terminal_reason: str


def _project(z: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(z, lower), upper)


def _boundary_alpha(step: np.ndarray, direction: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    alpha = np.inf
    positive = direction > 0.0
    negative = direction < 0.0
    if np.any(positive):
        alpha = min(alpha, float(np.min((upper[positive] - step[positive]) / direction[positive])))
    if np.any(negative):
        alpha = min(alpha, float(np.min((lower[negative] - step[negative]) / direction[negative])))
    if alpha == 0.0:
        return np.inf
    return float(alpha) if np.isfinite(alpha) else np.inf


def _radius_alpha(step: np.ndarray, direction: np.ndarray, radius: float) -> float:
    a = float(direction @ direction)
    if a <= 0.0:
        return np.inf
    b = 2.0 * float(step @ direction)
    c = float(step @ step) - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        discriminant = 0.0
    if abs(b) <= 1.0e-14:
        return max(0.0, float(np.sqrt(-c / a)))
    if b >= 0.0:
        root = (2.0 * -c) / (b + np.sqrt(discriminant))
    else:
        root = (-b + np.sqrt(discriminant)) / (2.0 * a)
    if root < -1.0e-12:
        return np.inf
    return max(0.0, float(root))


def solve_boundary_trust_region(
    hessian: np.ndarray,
    gradient: np.ndarray,
    current: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    radius: float,
    cg_tolerance: float = 1.0e-10,
) -> TrustRegionStep:
    """Minimize the local quadratic over box and trust ball.

    Coordinates reaching a finite bound are fixed and CG is restarted.  A
    trust-radius intersection terminates with a boundary-limited step.
    """

    p = len(gradient)
    if p == 0:
        return TrustRegionStep(np.zeros(0), 0.0, 0, False, "no_parameters")

    step = np.zeros(p)
    cg_stop_tolerance = cg_tolerance * max(1.0, float(np.linalg.norm(gradient)))
    fixed = np.zeros(p, dtype=bool)
    fixed |= np.isfinite(lower) & (np.abs(current - lower) <= 1.0e-12) & (gradient > 0.0)
    fixed |= np.isfinite(upper) & (np.abs(current - upper) <= 1.0e-12) & (gradient < 0.0)
    total_iterations = 0

    restart = 0
    while restart <= p:
        restart += 1
        current_gradient = gradient + hessian @ step
        current_point = current + step
        fixed = np.zeros(p, dtype=bool)
        fixed |= np.isfinite(lower) & (np.abs(current_point - lower) <= 1.0e-12) & (current_gradient >= 0.0)
        fixed |= np.isfinite(upper) & (np.abs(current_point - upper) <= 1.0e-12) & (current_gradient <= 0.0)
        free = ~fixed
        if not np.any(free):
            break

        residual = current_gradient.copy()
        residual[fixed] = 0.0
        if float(np.linalg.norm(residual)) <= cg_stop_tolerance:
            break

        direction = -residual
        residual_sq = float(residual @ residual)

        for _ in range(2 * p + 1):
            total_iterations += 1
            h_direction = hessian @ direction
            h_direction[fixed] = 0.0
            curvature = float(direction @ h_direction)
            free_direction = direction.copy()
            free_direction[fixed] = 0.0
            alpha_boundary = _boundary_alpha(current + step, free_direction, lower, upper)
            alpha_radius = _radius_alpha(step, free_direction, radius)

            if curvature <= 1.0e-14:
                alpha = min(alpha_boundary, alpha_radius)
                if not np.isfinite(alpha) or alpha <= 0.0:
                    break
                step = step + alpha * free_direction
                if alpha >= alpha_radius - 1.0e-12:
                    return _result(step, gradient, hessian, total_iterations, True, "negative_curvature_trust_boundary")
                candidate = current + step
                fixed |= np.isfinite(lower) & (np.abs(candidate - lower) <= 1.0e-10)
                fixed |= np.isfinite(upper) & (np.abs(candidate - upper) <= 1.0e-10)
                break

            alpha_cg = residual_sq / curvature
            alpha = min(alpha_cg, alpha_boundary, alpha_radius)
            if not np.isfinite(alpha) or alpha <= 0.0:
                break

            step = step + alpha * free_direction
            if alpha >= alpha_boundary - 1.0e-12:
                candidate = current + step
                hit_lower = np.isfinite(lower) & (np.abs(candidate - lower) <= 1.0e-10)
                hit_upper = np.isfinite(upper) & (np.abs(candidate - upper) <= 1.0e-10)
                current_gradient = gradient + hessian @ step
                fixed |= hit_lower & (current_gradient >= 0.0)
                fixed |= hit_upper & (current_gradient <= 0.0)
                if fixed.all():
                    return _result(step, gradient, hessian, total_iterations, False, "parameter_boundary")
                break

            if alpha >= alpha_radius - 1.0e-12:
                return _result(step, gradient, hessian, total_iterations, True, "trust_boundary")

            residual = residual + alpha * h_direction
            residual[fixed] = 0.0
            if float(np.linalg.norm(residual)) <= cg_stop_tolerance:
                return _result(step, gradient, hessian, total_iterations, False, "cg_converged")

            new_sq = float(residual @ residual)
            beta = new_sq / max(residual_sq, 1.0e-300)
            direction = -residual + beta * direction
            direction[fixed] = 0.0
            residual_sq = new_sq

    predicted = -float(gradient @ step) - 0.5 * float(step @ hessian @ step)
    if predicted > 1.0e-14:
        return TrustRegionStep(step, predicted, total_iterations, False, "parameter_boundary")

    # A Cauchy step is a defensive fallback for a degenerate active restart.
    direction = _project(-gradient, lower - current, upper - current)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return TrustRegionStep(np.zeros(p), 0.0, total_iterations, False, "stationary")
    alpha = min(radius / norm, _boundary_alpha(np.zeros(p), direction, lower - current, upper - current))
    curvature = float(direction @ hessian @ direction)
    if curvature > 0.0:
        alpha = min(alpha, norm * norm / curvature)
    fallback_step = alpha * direction
    predicted = -float(gradient @ fallback_step) - 0.5 * float(fallback_step @ hessian @ fallback_step)
    return TrustRegionStep(fallback_step, predicted, total_iterations + 1, alpha >= radius / norm, "cauchy_fallback")


def _result(
    step: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    iterations: int,
    hit_trust: bool,
    reason: str,
) -> TrustRegionStep:
    predicted = -float(gradient @ step) - 0.5 * float(step @ hessian @ step)
    return TrustRegionStep(step, predicted, iterations, hit_trust, reason)
