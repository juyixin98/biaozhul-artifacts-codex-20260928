"""Run local fixtures and compare production results to the exact reference."""

from __future__ import annotations

import argparse
import json
import platform
from fractions import Fraction
from pathlib import Path

import numpy
import scipy

from core import __version__
from core.contracts import AnalysisRequest
from core.metrics import NumpySamples, compute_analysis
from core.repository import ALGORITHM_VERSION
from .fixtures import generate_synthetic_cases
from .reference import reference_from_fixture


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_data"
DEFAULT_REPORT = ROOT / "reports" / "experiment_report.json"


def fraction_to_float(value: object) -> float:
    return float(value) if isinstance(value, Fraction) else value


def load_hand_fixtures() -> list[dict]:
    fixtures = []
    for path in sorted(SAMPLE_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            fixture = json.load(handle)
        fixture["fixture_path"] = str(path.relative_to(ROOT))
        fixtures.append(fixture)
    return fixtures


def production_result(fixture: dict):
    samples = fixture["samples"]
    request = AnalysisRequest(
        y=samples["y"],
        p=samples["p"],
        w=samples.get("w"),
        method=fixture["method"],
        n_bins=fixture.get("n_bins", 10),
        edges=fixture.get("edges"),
        run_label=fixture["name"],
    )
    return compute_analysis(
        NumpySamples.from_contract(request),
        method=request.method,
        requested_bins=request.n_bins,
        run_id="experiment-" + fixture["name"],
        explicit_edges=request.edges,
    )


def compare_metrics(result, reference: dict, expected: dict | None = None) -> tuple[bool, list[dict], list[str]]:
    checks = []
    steps = []
    metric_names = [
        "total_weight",
        "ess",
        "base_rate",
        "mean_prediction",
        "raw_brier",
        "binned_brier",
        "reliability",
        "resolution",
        "uncertainty",
        "decomposition_residual",
        "identity_residual",
        "sum_weight_fractions",
    ]
    production_map = {
        "total_weight": result.weights.total_weight,
        "ess": result.weights.effective_sample_size,
        "base_rate": result.base_rate,
        "mean_prediction": result.mean_prediction,
        "raw_brier": result.decomposition.raw_brier,
        "binned_brier": result.decomposition.binned_brier,
        "reliability": result.decomposition.reliability,
        "resolution": result.decomposition.resolution,
        "uncertainty": result.decomposition.uncertainty,
        "decomposition_residual": result.decomposition.decomposition_residual,
        "identity_residual": result.decomposition.identity_residual,
        "sum_weight_fractions": result.weights.sum_weight_fractions,
    }
    for name in metric_names:
        expected_value = float(reference[name])
        actual_value = float(production_map[name])
        absolute_error = abs(actual_value - expected_value)
        passed = absolute_error < 1e-12
        checks.append(
            {
                "metric": name,
                "expected_exact": str(reference[name]),
                "actual": actual_value,
                "absolute_error": absolute_error,
                "passed": passed,
                "basis": "independent Fraction reference; tolerance 1e-12",
            }
        )
        steps.append(f"{name}: reference={reference[name]} actual={actual_value:.15g} pass={passed}")

    edges_match = [float(value) for value in result.edges] == [float(value) for value in reference["edges"]]
    checks.append({"metric": "edges", "expected_exact": [str(value) for value in reference["edges"]], "actual": result.edges, "absolute_error": 0.0, "passed": edges_match, "basis": "tie-safe deterministic boundaries"})
    if expected:
        alias = {"effective_sample_size": "ess"}
        for key, value in expected.items():
            if key in alias:
                actual_value = production_map[alias[key]]
            elif key == "zero_weight_count":
                actual_value = result.weights.count and sum(item.zero_weight_count for item in result.bins)
            elif hasattr(result.weights, key):
                actual_value = getattr(result.weights, key)
            elif hasattr(result.decomposition, key):
                actual_value = getattr(result.decomposition, key)
            elif hasattr(result, key):
                actual_value = getattr(result, key)
            else:
                continue
            passed = abs(float(actual_value) - float(value)) < 1e-10
            checks.append(
                {
                    "metric": "fixture_expected." + key,
                    "expected_exact": value,
                    "actual": actual_value,
                    "absolute_error": abs(float(actual_value) - float(value)),
                    "passed": passed,
                    "basis": "hand-authored fixture expectation; tolerance 1e-10",
                }
            )
            steps.append(f"hand expectation {key}: expected={value} actual={actual_value} pass={passed}")
    return all(item["passed"] for item in checks), checks, steps


def evaluate_case(fixture: dict) -> dict:
    reference = reference_from_fixture(fixture)
    result = production_result(fixture)
    passed, checks, steps = compare_metrics(result, reference, fixture.get("expected"))
    occupied_match = result.occupied_bins == sum(1 for item in reference["bins"] if not item["empty"])
    empty_match = result.empty_bins == sum(1 for item in reference["bins"] if item["empty"])
    passed = passed and occupied_match and empty_match
    steps.append(f"occupied bins production={result.occupied_bins} reference={sum(1 for item in reference['bins'] if not item['empty'])}")
    steps.append(f"empty bins production={result.empty_bins} reference={sum(1 for item in reference['bins'] if item['empty'])}")
    return {
        "name": fixture["name"],
        "status": "success" if passed else "failed",
        "fixture_path": fixture.get("fixture_path"),
        "source": fixture.get("source", fixture.get("fixture_path", "static hand fixture")),
        "steps": steps,
        "checks": checks,
        "production_edges": result.edges,
        "occupied_bins": result.occupied_bins,
        "empty_bins": result.empty_bins,
        "trace_excerpt": result.calculation_trace[:4],
    }


def run(include_synthetic: bool = True) -> dict:
    fixtures = load_hand_fixtures()
    if include_synthetic:
        for fixture in generate_synthetic_cases():
            fixture.setdefault("fixture_path", "generated")
            fixtures.append(fixture)

    cases = [evaluate_case(fixture) for fixture in fixtures]
    passed = all(case["status"] == "success" for case in cases)
    return {
        "run_identity": "experiment-report-" + str(abs(hash(json.dumps(cases, sort_keys=True, default=str))) % (10**12)),
        "overall_status": "success" if passed else "failed",
        "summary": {
            "total_cases": len(cases),
            "passed_cases": sum(case["status"] == "success" for case in cases),
            "failed_cases": sum(case["status"] != "success" for case in cases),
        },
        "versions": {
            "python": platform.python_version(),
            "app": __version__,
            "algorithm": ALGORITHM_VERSION,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-synthetic", action="store_true")
    args = parser.parse_args()
    report = run(include_synthetic=not args.no_synthetic)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=fraction_to_float), encoding="utf-8")
    print(json.dumps({"overall_status": report["overall_status"], "summary": report["summary"], "report": str(args.report)}, indent=2))
    return 0 if report["overall_status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
