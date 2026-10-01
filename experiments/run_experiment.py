from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.oracle import beta_posterior, normalized_beta_binomial


def scenario(name: str, alpha: float, beta: float, successes: int, failures: int, n: int) -> dict:
    posterior = beta_posterior(alpha, beta, successes, failures)
    probabilities, probability_sum = normalized_beta_binomial(
        n, posterior.alpha, posterior.beta
    )
    return {
        "name": name,
        "input": {
            "prior_alpha": alpha,
            "prior_beta": beta,
            "successes": successes,
            "failures": failures,
            "predictive_trials": n,
        },
        "posterior": {
            "alpha": posterior.alpha,
            "beta": posterior.beta,
            "mean": posterior.mean,
            "variance": posterior.variance,
        },
        "predictive": {
            "probability_sum": probability_sum,
            "all_finite": bool(np.all(np.isfinite(probabilities))),
            "nonnegative": bool(np.all(probabilities >= 0.0)),
            "first_five": [float(x) for x in probabilities[: min(5, n + 1)]],
        },
        "judgement": "Probability mass must be finite, non-negative, and sum to one.",
    }


def run() -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": [
            scenario("hand_small_counts", 1.0, 1.0, 2, 1, 2),
            scenario("all_success", 1.0, 1.0, 10_000, 0, 20),
            scenario("large_counts", 0.5, 0.5, 10**12, 10**12, 200),
            scenario("extreme_prior", 1.0e100, 1.0, 1_000_000, 0, 10),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = run()
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

