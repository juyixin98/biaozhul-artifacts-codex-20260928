"""Local synthetic equipment-lifetime data generator.

The model is deliberately simple: Weibull equipment failure times plus
independent inspection censor times. It exists for exercising the API and
tests, not for inferring real component reliability.
"""

from __future__ import annotations

import numpy as np

from .contracts import SyntheticDatasetRequest
from .tracing import trace_entry

GROUP_PARAMETERS = {
    "A": {"shape": 2.2, "scale": 8.0},
    "B": {"shape": 1.8, "scale": 11.0},
    "C": {"shape": 2.6, "scale": 14.0},
}


def generate_dataset(request: SyntheticDatasetRequest) -> dict:
    rng = np.random.default_rng(request.seed)
    observations: list[dict] = []

    for group, parameters in GROUP_PARAMETERS.items():
        for index in range(request.sample_size_per_group):
            lifetime = float(
                rng.weibull(parameters["shape"]) * parameters["scale"]
            )
            censor_time = float(
                rng.uniform(request.censor_low, request.censor_high)
            )
            lifetime = round(max(0.1, lifetime), 1)
            censor_time = round(censor_time, 1)
            if censor_time == lifetime:
                censor_time = round(censor_time + 0.1, 1)

            event = lifetime < censor_time
            terminal_time = min(lifetime, censor_time)
            entry = 0.0
            if request.delayed_entry:
                upper = min(2.0, max(0.0, terminal_time - 0.2))
                entry = round(float(rng.uniform(0.0, upper)) if upper > 0 else 0.0, 1)

            observations.append(
                {
                    "observation_id": f"{group}-{index + 1:04d}",
                    "group": group,
                    "time": terminal_time,
                    "event": event,
                    "entry": entry,
                }
            )

    trace = [
        trace_entry(
            "synthetic-generator",
            "weibull_equipment_lifetimes",
            distribution="weibull",
            group_parameters=GROUP_PARAMETERS,
            censor_distribution="uniform",
            delayed_entry=request.delayed_entry,
            seed=request.seed,
        )
    ]
    return {
        "generator_version": "synthetic-equipment-weibull.v1",
        "purpose": "synthetic local fixture only; not a real medical or reliability conclusion",
        "groups": list(GROUP_PARAMETERS),
        "observations": observations,
        "trace": trace,
    }
