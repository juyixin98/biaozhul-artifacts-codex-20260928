"""Run synthetic scenarios and emit a versioned reproducibility report."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from changepoint_api.experiments.fixtures import (
    Fixture,
    mean_step,
    single_outlier,
    tie_fixture,
)
from changepoint_api.experiments.oracle import oracle_optimum
from changepoint_api.kernel import detect_mean_changepoints
from changepoint_api.versions import (
    ALGORITHM_VERSION,
    PACKAGE_VERSION,
    TIE_BREAK_RULE,
)


def _scenarios() -> list[tuple[Fixture, float, int]]:
    return [
        (mean_step(), 0.5, 2),
        (single_outlier(), 2.0, 1),
        (tie_fixture(), 0.0, 1),
        (tie_fixture(), 1.0e6, 1),
    ]


def _result_payload(result) -> dict:
    return {
        "change_points": result.change_points,
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "length": segment.length,
                "mean": segment.mean,
                "cost": segment.cost,
            }
            for segment in result.segments
        ],
        "total_sse": result.total_sse,
        "objective": result.objective,
        "candidate_evaluations": result.candidate_evaluations,
        "diagnostics": result.diagnostics,
    }


def build_report() -> dict:
    scenario_reports = []
    for fixture, penalty, min_length in _scenarios():
        exact = detect_mean_changepoints(
            fixture.samples,
            penalty=penalty,
            min_segment_length=min_length,
            algorithm="exact_dp",
        )
        pruned = detect_mean_changepoints(
            fixture.samples,
            penalty=penalty,
            min_segment_length=min_length,
            algorithm="pruned_dp",
        )
        oracle = oracle_optimum(
            fixture.samples, penalty, min_length
        )
        exact_matches_pruned = (
            exact.objective == pruned.objective
            and exact.change_points == pruned.change_points
            and [segment.cost for segment in exact.segments]
            == [segment.cost for segment in pruned.segments]
        )
        exact_matches_oracle = (
            math_isclose(exact.objective, oracle.objective)
            and tuple(exact.change_points) == oracle.change_points
        )
        scenario_reports.append(
            {
                "fixture": asdict(fixture),
                "penalty": penalty,
                "min_segment_length": min_length,
                "exact": _result_payload(exact),
                "pruned": _result_payload(pruned),
                "independent_oracle": {
                    "change_points": list(oracle.change_points),
                    "segments": [list(part) for part in oracle.segments],
                    "costs": list(oracle.costs),
                    "total_sse": oracle.total_sse,
                    "objective": oracle.objective,
                },
                "judgements": {
                    "exact_matches_pruned": exact_matches_pruned,
                    "exact_matches_independent_oracle": exact_matches_oracle,
                    "oracle_method": (
                        "independent exhaustive recursion with Welford SSE"
                    ),
                },
            }
        )

    return {
        "package_version": PACKAGE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "tie_break_rule": TIE_BREAK_RULE,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scenarios": scenario_reports,
    }


def math_isclose(left: float, right: float, rel_tol: float = 1e-12) -> bool:
    return abs(left - right) <= rel_tol * max(1.0, abs(left), abs(right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiment-report.json"),
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    failed = [
        scenario["fixture"]["name"]
        for scenario in report["scenarios"]
        if not all(scenario["judgements"].values())
    ]
    if failed:
        raise SystemExit(f"experiment judgements failed: {failed}")
    print(f"wrote {args.output} with {len(report['scenarios'])} scenarios")


if __name__ == "__main__":
    main()
