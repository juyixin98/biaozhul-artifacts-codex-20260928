"""Explicit parameter normalization for bounded trust-region steps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterScaling:
    """Map every physical parameter to finite normalized trust coordinates.

    For a doubly bounded parameter, coordinates are in ``[0, 1]``.  A single
    finite bound is represented by a one-sided ``asinh`` map anchored at that
    bound.  Unbounded parameters use ``theta = scale * z``.  All trust radii
    therefore refer to dimensionless parameter steps.
    """

    lower: np.ndarray
    upper: np.ndarray
    scales: np.ndarray

    @classmethod
    def create(cls, lower: np.ndarray, upper: np.ndarray, scales: np.ndarray) -> "ParameterScaling":
        lower_array = np.asarray(lower, dtype=float)
        upper_array = np.asarray(upper, dtype=float)
        scales_array = np.asarray(scales, dtype=float)
        if np.any(lower_array >= upper_array):
            raise ValueError("lower bounds must be strictly below upper bounds")
        if np.any(scales_array <= 0.0) or not np.all(np.isfinite(scales_array)):
            raise ValueError("parameter scales must be finite and positive")
        return cls(lower_array, upper_array, scales_array)

    @property
    def n(self) -> int:
        return len(self.lower)

    def internal_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.n, -np.inf, dtype=float)
        upper = np.full(self.n, np.inf, dtype=float)
        both = np.isfinite(self.lower) & np.isfinite(self.upper)
        at_lower = np.isfinite(self.lower) & ~np.isfinite(self.upper)
        at_upper = np.isfinite(self.upper) & ~np.isfinite(self.lower)
        lower[both | at_lower] = 0.0
        upper[both] = 1.0
        upper[at_upper] = 0.0
        return lower, upper

    def to_internal(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        z = np.zeros(self.n)
        both = np.isfinite(self.lower) & np.isfinite(self.upper)
        at_lower = np.isfinite(self.lower) & ~np.isfinite(self.upper)
        at_upper = np.isfinite(self.upper) & ~np.isfinite(self.lower)
        free = ~np.isfinite(self.lower) & ~np.isfinite(self.upper)

        if np.any(both):
            span = self.upper[both] - self.lower[both]
            q = np.arcsinh(span / self.scales[both])
            z[both] = np.arcsinh((theta[both] - self.lower[both]) / self.scales[both]) / q
        if np.any(at_lower):
            z[at_lower] = np.arcsinh((theta[at_lower] - self.lower[at_lower]) / self.scales[at_lower])
        if np.any(at_upper):
            z[at_upper] = np.arcsinh((self.upper[at_upper] - theta[at_upper]) / self.scales[at_upper])
        if np.any(free):
            z[free] = theta[free] / self.scales[free]
        return z

    def to_external(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        theta = np.zeros(self.n)
        both = np.isfinite(self.lower) & np.isfinite(self.upper)
        at_lower = np.isfinite(self.lower) & ~np.isfinite(self.upper)
        at_upper = np.isfinite(self.upper) & ~np.isfinite(self.lower)
        free = ~np.isfinite(self.lower) & ~np.isfinite(self.upper)

        if np.any(both):
            span = self.upper[both] - self.lower[both]
            q = np.arcsinh(span / self.scales[both])
            theta[both] = self.lower[both] + self.scales[both] * np.sinh(q * z[both])
        if np.any(at_lower):
            theta[at_lower] = self.lower[at_lower] + self.scales[at_lower] * np.sinh(z[at_lower])
        if np.any(at_upper):
            theta[at_upper] = self.upper[at_upper] - self.scales[at_upper] * np.sinh(z[at_upper])
        if np.any(free):
            theta[free] = self.scales[free] * z[free]
        return theta

    def external_gradient_from_internal(self, z: np.ndarray) -> np.ndarray:
        """Return ``dtheta_i / dz_i`` for transforming physical Jacobians."""

        z = np.asarray(z, dtype=float)
        derivative = np.ones(self.n)
        both = np.isfinite(self.lower) & np.isfinite(self.upper)
        at_lower = np.isfinite(self.lower) & ~np.isfinite(self.upper)
        at_upper = np.isfinite(self.upper) & ~np.isfinite(self.lower)
        free = ~np.isfinite(self.lower) & ~np.isfinite(self.upper)

        if np.any(both):
            span = self.upper[both] - self.lower[both]
            q = np.arcsinh(span / self.scales[both])
            derivative[both] = self.scales[both] * q * np.cosh(q * z[both])
        if np.any(at_lower):
            derivative[at_lower] = self.scales[at_lower] * np.cosh(z[at_lower])
        if np.any(at_upper):
            derivative[at_upper] = self.scales[at_upper] * np.cosh(z[at_upper])
        if np.any(free):
            derivative[free] = self.scales[free]
        return derivative
