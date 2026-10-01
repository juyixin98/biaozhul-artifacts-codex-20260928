"""Deterministic local reproducibility experiment.

The experiment compares the application posterior against direct Beta algebra
and compares predictive output against scipy.stats.betabinom. It uses fixed
integer fixtures, so no random seed is required.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

from app.service import BetaBinomialService
from app.storage import SQLiteStore


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "experiments" / "fixtures" / "experiment_cases.json"


def _close(a: float, b: float, rel_tol: float = 1e-12, abs_tol: float = 1e-14) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _reference_predictions(future_n: int, alpha: float, beta: float) -> list[float]:
    k = np.arange(future_n + 1)
    probabilities = stats.betabinom.pmf(k, future_n, alpha, beta)
    return [float(x) for x in probabilities]


def run(output_dir: Path) -> dict:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "experiment-fixed-run"
    checks: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="beta-binomial-") as tmp:
        db_path = Path(tmp) / "experiment.sqlite3"
        service = BetaBinomialService(SQLiteStore(db_path))
        service.create_run(
            run_id, fixture["prior_alpha"], fixture["prior_beta"]
        )

        split_post = None
        for item in fixture["split_case"]:
            result = service.submit_batch(
                run_id, item["batch_id"], item["successes"], item["failures"]
            )
            split_post = (result.posterior_alpha, result.posterior_beta)
        assert split_post is not None
        combined_alpha = fixture["prior_alpha"] + fixture["combined_case"]["successes"]
        combined_beta = fixture["prior_beta"] + fixture["combined_case"]["failures"]
        checks.append(
            {
                "name": "split_equals_combined_posterior",
                "reason": "Beta sufficient statistics are additive.",
                "observed": {"alpha": split_post[0], "beta": split_post[1]},
                "expected": {"alpha": combined_alpha, "beta": combined_beta},
                "passed": _close(split_post[0], combined_alpha)
                and _close(split_post[1], combined_beta),
            }
        )

        prediction = service.predict(
            run_id, fixture["future_n"], confidence_level=0.95
        )
        reference = _reference_predictions(
            fixture["future_n"], *split_post
        )
        observed = [point.probability for point in prediction.points]
        max_abs_error = max(abs(a - b) for a, b in zip(observed, reference))
        checks.append(
            {
                "name": "prediction_matches_scipy_betabinom",
                "reason": "Reference is scipy.stats.betabinom, not app.stats.",
                "observed_sum": sum(observed),
                "max_abs_error": max_abs_error,
                "passed": _close(sum(observed), 1.0, abs_tol=1e-15)
                and max_abs_error < 1e-12,
            }
        )

        revision = fixture["revision"]
        revised = service.revise_batch(
            run_id,
            revision["batch_id"],
            revision["to"]["successes"],
            revision["to"]["failures"],
            reason="fixed deterministic correction",
        )
        expected = revision["expected_after_revision"]
        checks.append(
            {
                "name": "revision_withdraws_old_sufficient_statistics",
                "reason": "Old active batch row is superseded and only corrected counts remain.",
                "observed": {
                    "successes": revised.total_successes,
                    "failures": revised.total_failures,
                },
                "expected": expected,
                "passed": revised.total_successes == expected["successes"]
                and revised.total_failures == expected["failures"],
            }
        )

        versions = service.list_versions(run_id)
        checks.append(
            {
                "name": "revision_creates_versioned_result",
                "reason": "Old rows remain; run sequence records created, submit, submit, revise.",
                "observed_sequence": [v.event_type for v in versions],
                "expected_sequence": [
                    "created",
                    "batch_submitted",
                    "batch_submitted",
                    "batch_revised",
                ],
                "passed": [v.event_type for v in versions]
                == [
                    "created",
                    "batch_submitted",
                    "batch_submitted",
                    "batch_revised",
                ],
            }
        )

    large = fixture["large_case"]
    service = BetaBinomialService(SQLiteStore(":memory:"))
    service.create_run("large-case", fixture["prior_alpha"], fixture["prior_beta"])
    result = service.submit_batch(
        "large-case",
        "large",
        large["successes"],
        large["failures"],
    )
    large_prediction = service.predict("large-case", 100)
    checks.append(
        {
            "name": "large_counts_remain_finite",
            "reason": "Posterior and full prediction are finite after large counts.",
            "observed": {
                "alpha": result.posterior_alpha,
                "beta": result.posterior_beta,
                "prediction_sum": large_prediction.probability_sum,
            },
            "passed": math.isfinite(result.posterior_alpha)
            and math.isfinite(result.posterior_beta)
            and _close(large_prediction.probability_sum, 1.0, abs_tol=1e-15),
        }
    )

    import fastapi
    import numpy
    import pydantic
    import pytest
    import scipy

    report = {
        "run_number": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_path": str(FIXTURE_PATH),
        "fixture": fixture,
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                "fastapi": fastapi.__version__,
                "numpy": numpy.__version__,
                "pydantic": pydantic.__version__,
                "pytest": pytest.__version__,
                "scipy": scipy.__version__,
            },
        },
    }

    report_path = output_dir / "experiment_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "experiments",
    )
    args = parser.parse_args()
    report = run(args.output_dir)
    print(json.dumps({
        "run_number": report["run_number"],
        "all_passed": report["all_passed"],
        "output": str(args.output_dir / "experiment_report.json"),
    }, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
