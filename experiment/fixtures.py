"""Deterministic synthetic cases generated without external services."""

from __future__ import annotations

import numpy as np
from scipy import stats


def generate_synthetic_cases(seed: int = 320) -> list[dict]:
    rng = np.random.default_rng(seed)

    # Beta forecasts, Bernoulli outcomes conditionally generated from forecasts.
    well_cal_p = stats.beta.rvs(3, 5, size=40, random_state=rng).tolist()
    well_cal_y = [int(rng.random() < probability) for probability in well_cal_p]

    # Extremes around a balanced outcome process.
    over_y = rng.integers(0, 2, size=40).tolist()
    over_p = [0.05 if outcome == 0 else 0.95 for outcome in over_y]

    constant_p = [0.37] * 30
    constant_y = [int(rng.random() < 0.63) for _ in constant_p]

    return [
        {
            "name": "synthetic_beta_forecasts",
            "description": "Seeded SciPy beta forecasts with Bernoulli outcomes.",
            "method": "quantile",
            "n_bins": 5,
            "samples": {"y": well_cal_y, "p": well_cal_p},
            "source": f"numpy_default_rng({seed}) + scipy.stats.beta.rvs(3, 5)",
        },
        {
            "name": "synthetic_overconfident",
            "description": "Seeded extreme forecasts with balanced outcomes.",
            "method": "fixed",
            "n_bins": 2,
            "samples": {"y": over_y, "p": over_p},
            "source": f"numpy_default_rng({seed})",
        },
        {
            "name": "synthetic_constant",
            "description": "Seeded constant forecasts with a different base rate.",
            "method": "fixed",
            "n_bins": 4,
            "samples": {"y": constant_y, "p": constant_p},
            "source": f"numpy_default_rng({seed})",
        },
    ]
