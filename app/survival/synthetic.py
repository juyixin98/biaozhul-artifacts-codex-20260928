"""Deterministic synthetic device-lifetime data generation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import ObservationIn
from ..versions import SYNTHETIC_GENERATOR_VERSION


class SyntheticDatasetRequest(BaseModel):
    seed: int = Field(default=311, ge=0, le=2**32 - 1)
    sample_size_per_group: int = Field(default=24, ge=2, le=500)
    event_scale_control: float = Field(default=12.0, gt=0)
    event_scale_treatment: float = Field(default=18.0, gt=0)
    censor_scale: float = Field(default=20.0, gt=0)
    include_left_truncation: bool = False
    max_entry_time: float = Field(default=3.0, ge=0)


class SyntheticDatasetMetadata(BaseModel):
    generator_version: str
    seed: int
    method: str
    groups: list[str]
    disclaimer: str


class SyntheticDatasetResponse(BaseModel):
    metadata: SyntheticDatasetMetadata
    observations: list[ObservationIn]


def generate_synthetic_dataset(request: SyntheticDatasetRequest) -> SyntheticDatasetResponse:
    """Generate right-censored exponential device lifetimes.

    Lifetimes and independent right-censoring times are sampled from
    exponential distributions. Delayed-entry times, when requested, are
    uniform and are rejected/redrawn if they do not precede observation time.
    """

    rng = np_random_generator(request.seed)
    observations: list[ObservationIn] = []
    scales = {
        "control": request.event_scale_control,
        "treatment": request.event_scale_treatment,
    }

    for group, scale in scales.items():
        for _ in range(request.sample_size_per_group):
            lifetime = float(rng.exponential(scale))
            censor_time = float(rng.exponential(request.censor_scale))
            observed_time = min(lifetime, censor_time)
            entry_time = 0.0
            if request.include_left_truncation:
                while entry_time <= 0.0 or entry_time >= observed_time:
                    entry_time = float(rng.uniform(0.0, request.max_entry_time))
                    if request.max_entry_time == 0:
                        break
            observations.append(
                ObservationIn(
                    time=round(max(observed_time, 1e-9), 6),
                    event=bool(lifetime <= censor_time),
                    group=group,
                    entry_time=round(entry_time, 6),
                )
            )

    return SyntheticDatasetResponse(
        metadata=SyntheticDatasetMetadata(
            generator_version=SYNTHETIC_GENERATOR_VERSION,
            seed=request.seed,
            method="independent exponential lifetimes and censoring; uniform delayed entry when enabled",
            groups=["control", "treatment"],
            disclaimer=(
                "Synthetic demonstration data only. Not a medical device "
                "safety claim and not medical advice."
            ),
        ),
        observations=observations,
    )


def np_random_generator(seed: int):
    # Isolated helper keeps generation explicit and easy to patch in tests.
    import numpy as np

    return np.random.default_rng(seed)
