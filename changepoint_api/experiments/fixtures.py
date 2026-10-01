"""Synthetic local data generators used by experiments and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fixture:
    name: str
    samples: list[float]
    expected_change_points: list[int]
    description: str


def mean_step(
    *,
    n_per_segment: int = 10,
    jump: float = 5.0,
    noise_sd: float = 0.1,
    seed: int = 316,
) -> Fixture:
    rng = np.random.default_rng(seed)
    first = rng.normal(0.0, noise_sd, n_per_segment)
    second = rng.normal(jump, noise_sd, n_per_segment)
    return Fixture(
        name="mean_step",
        samples=[float(x) for x in np.concatenate([first, second])],
        expected_change_points=[n_per_segment],
        description="Two means with small independent Gaussian noise.",
    )


def single_outlier(
    *,
    n: int = 20,
    outlier_index: int = 10,
    outlier_value: float = 30.0,
    seed: int = 317,
) -> Fixture:
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, 0.05, n)
    samples[outlier_index] = outlier_value
    return Fixture(
        name="single_outlier",
        samples=[float(x) for x in samples],
        expected_change_points=[],
        description=(
            "Constant mean with one severe outlier; a moderate penalty should "
            "not turn one point into a persistent step."
        ),
    )


def tie_fixture() -> Fixture:
    return Fixture(
        name="deterministic_tie",
        samples=[0.0, 0.0, 1.0, 1.0],
        expected_change_points=[2],
        description="Symmetric constant blocks with an exact two-block tie.",
    )
