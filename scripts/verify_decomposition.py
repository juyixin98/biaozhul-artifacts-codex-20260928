#!/usr/bin/env python3
"""Run the local reproducible STL verification experiment."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stl_backend.config import DEFAULT_CONFIG
from stl_backend.kernel import stl_decompose


DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "robust_stl.json"


def verify(fixture_path: pathlib.Path) -> dict:
    fixture = json.loads(fixture_path.read_text("utf-8"))
    observed = np.asarray(fixture["observed"], dtype=np.float64)
    result = stl_decompose(observed, DEFAULT_CONFIG)
    reference = fixture["reference"]

    comparisons = {}
    for name in ("trend", "seasonal", "residual", "robust_weights"):
        actual = np.asarray(result[{
            "trend": "trend",
            "seasonal": "seasonal",
            "residual": "residual",
            "robust_weights": "robust_weights",
        }[name]])
        expected = np.asarray(reference[name], dtype=np.float64)
        comparisons[name] = {
            "max_abs_difference": float(np.max(np.abs(actual - expected))),
            "mean_abs_difference": float(np.mean(np.abs(actual - expected))),
        }

    reconstruction = (
        result["trend"] + result["seasonal"] + result["residual"]
    )
    checks = {
        "fixed_period": DEFAULT_CONFIG.period == 7,
        "fixed_seasonal_window": DEFAULT_CONFIG.seasonal_window == 7,
        "fixed_trend_window": DEFAULT_CONFIG.trend_window == 15,
        "fixed_low_pass_window": DEFAULT_CONFIG.low_pass_window == 9,
        "seasonal_centered": abs(float(np.mean(result["seasonal"]))) < 1e-12,
        "additive_reconstruction": bool(
            np.max(np.abs(observed - reconstruction)) < 1e-12
        ),
        "known_outliers_zero_weighted": all(
            result["robust_weights"][index] < 1e-8
            for index in fixture["outlier_positions"]
        ),
        "matches_reference_within_5e_10": all(
            item["max_abs_difference"] < 5e-10 for item in comparisons.values()
        ),
        "finite_components": all(
            bool(np.all(np.isfinite(result[name])))
            for name in ("trend", "seasonal", "residual", "robust_weights")
        ),
    }
    return {
        "fixture": str(fixture_path),
        "reference_implementation": fixture["reference_implementation"],
        "kernel_version": result["kernel_version"],
        "observations": int(observed.size),
        "checks": checks,
        "all_passed": all(checks.values()),
        "comparisons": comparisons,
        "diagnostics": {
            "seasonal_mean": float(np.mean(result["seasonal"])),
            "reconstruction_max_abs_error": float(
                np.max(np.abs(observed - reconstruction))
            ),
            "zero_weight_count": int(np.sum(result["robust_weights"] == 0.0)),
            "outlier_positions": fixture["outlier_positions"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        default=ROOT / "reports" / "verification.json",
    )
    args = parser.parse_args()

    report = verify(args.fixture)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
