"""Run deterministic synthetic experiments and report concrete comparisons."""

from __future__ import annotations

import json
import sys
from typing import Any

from experiments.reference import reference_detrended_pearson, scenarios
from lag_correlation.kernel.analysis import analyze_lag_correlation
from lag_correlation.kernel.models import Series


def run_all() -> tuple[list[dict[str, Any]], bool]:
    reports: list[dict[str, Any]] = []
    all_passed = True

    for scenario in scenarios():
        result = analyze_lag_correlation(
            x=Series(scenario.x_name, scenario.x_times, scenario.x_values),
            y=Series(scenario.y_name, scenario.y_times, scenario.y_values),
            min_lag=scenario.lag_grid[0],
            max_lag=scenario.lag_grid[1],
            step=scenario.lag_grid[2],
            tolerance=scenario.tolerance,
            request_id=f"experiment-{scenario.name}",
        )
        best = result.best_lag
        first_pair = best.pair_details[0].to_dict() if best and best.pair_details else {}
        independent_reference = _independent_reference(scenario.name, scenario)

        checks = {
            "status": result.status == scenario.expected_status,
            "failure_category": result.failure_category
            == scenario.expected_failure_category,
            "peak_lag": result.peak_lag == scenario.expected_peak_lag,
            "peak_correlation": _close(
                result.peak_correlation, scenario.expected_peak_correlation
            ),
            "anchor_side": result.anchor_side == scenario.expected_anchor_side,
            "n_pairs": (best.n_pairs if best else 0)
            == scenario.expected_best_n_pairs,
            "dense_used": (best.dense_samples_used if best else 0)
            == scenario.expected_dense_used,
            "dense_unmatched": (best.dense_samples_unmatched if best else 0)
            == scenario.expected_dense_unmatched,
            "first_pair": first_pair == scenario.expected_first_pair,
        }
        if independent_reference is not None and result.peak_correlation is not None:
            checks["independent_reference_correlation"] = abs(
                result.peak_correlation - independent_reference
            ) < 1e-12

        passed = all(checks.values())
        all_passed = all_passed and passed
        reports.append(
            {
                "scenario": scenario.name,
                "passed": passed,
                "checks": checks,
                "actual": {
                    "status": result.status,
                    "failure_category": result.failure_category,
                    "peak_lag": result.peak_lag,
                    "peak_correlation": result.peak_correlation,
                    "anchor_side": result.anchor_side,
                    "n_pairs": best.n_pairs if best else 0,
                    "dense_used": best.dense_samples_used if best else 0,
                    "dense_unmatched": best.dense_samples_unmatched if best else 0,
                    "first_pair": first_pair,
                },
                "independent_reference_correlation": independent_reference,
            }
        )

    return reports, all_passed


def main() -> int:
    reports, passed = run_all()
    print(json.dumps({"passed": passed, "experiments": reports}, indent=2, sort_keys=True))
    return 0 if passed else 1


def _close(actual: float | None, expected: float | None, tolerance: float = 1e-12) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return abs(actual - expected) <= tolerance


def _independent_reference(scenario_name: str, scenario) -> float | None:
    if scenario_name == "uniform_sampling_special_case":
        return reference_detrended_pearson(
            scenario.x_times[2:],
            scenario.x_values[2:],
            scenario.y_values[:8],
        )
    if scenario_name == "known_offset_different_density":
        dense_means = [
            sum(scenario.y_values[index : index + 3]) / 3.0
            for index in range(0, len(scenario.y_values), 3)
        ]
        return reference_detrended_pearson(scenario.x_times, scenario.x_values, dense_means)
    if scenario_name == "known_offset_with_long_gap":
        return reference_detrended_pearson(scenario.x_times, scenario.x_values, scenario.y_values)
    if scenario_name == "constant_after_detrend":
        return reference_detrended_pearson(scenario.x_times, scenario.x_values, scenario.y_values)
    return None


if __name__ == "__main__":
    sys.exit(main())
