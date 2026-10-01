import json
from fractions import Fraction
from pathlib import Path

import pytest

from core.contracts import AnalysisRequest
from core.metrics import NumpySamples, compute_analysis
from experiment.reference import reference_from_fixture
from tests.support import make_case_logger, read_jsonl

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "sample_data"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def analyze(fixture: dict):
    data = fixture["samples"]
    request = AnalysisRequest(
        y=data["y"],
        p=data["p"],
        w=data.get("w"),
        method=fixture["method"],
        n_bins=fixture.get("n_bins", 10),
        edges=fixture.get("edges"),
    )
    return compute_analysis(
        NumpySamples.from_contract(request),
        method=request.method,
        requested_bins=request.n_bins,
        run_id="unit-" + fixture["name"],
        explicit_edges=request.edges,
    )


@pytest.mark.parametrize(
    "filename",
    [
        "perfect_calibration.json",
        "overconfident.json",
        "constant_probability.json",
        "extreme_weights.json",
        "within_bin_residual.json",
        "quantile_ties.json",
        "empty_fixed_bins.json",
    ],
)
def test_fixture_matches_exact_reference_and_hand_expectations(filename, log_dir):
    fixture = load_fixture(filename)
    result = analyze(fixture)
    reference = reference_from_fixture(fixture)
    logger, log_path = make_case_logger(log_dir, fixture["name"])

    comparisons = {
        "total_weight": result.weights.total_weight,
        "ess": result.weights.effective_sample_size,
        "raw_brier": result.decomposition.raw_brier,
        "binned_brier": result.decomposition.binned_brier,
        "reliability": result.decomposition.reliability,
        "resolution": result.decomposition.resolution,
        "uncertainty": result.decomposition.uncertainty,
        "residual": result.decomposition.decomposition_residual,
        "identity_residual": result.decomposition.identity_residual,
        "sum_weight_fractions": result.weights.sum_weight_fractions,
    }
    reference_map = {
        "total_weight": reference["total_weight"],
        "ess": reference["ess"],
        "raw_brier": reference["raw_brier"],
        "binned_brier": reference["binned_brier"],
        "reliability": reference["reliability"],
        "resolution": reference["resolution"],
        "uncertainty": reference["uncertainty"],
        "residual": reference["decomposition_residual"],
        "identity_residual": reference["identity_residual"],
        "sum_weight_fractions": reference["sum_weight_fractions"],
    }

    for name, actual in comparisons.items():
        expected = float(reference_map[name])
        passed = abs(actual - expected) < 1e-12
        logger.log_case(
            input_digest=filename,
            expected=expected,
            actual=actual,
            passed=passed,
            basis=f"exact Fraction reference for {name}",
        )
        assert passed, (name, actual, expected)

    for key, value in fixture.get("expected", {}).items():
        if key == "zero_weight_count":
            actual = sum(item.zero_weight_count for item in result.bins)
        elif hasattr(result.weights, key):
            actual = getattr(result.weights, key)
        elif hasattr(result.decomposition, key):
            actual = getattr(result.decomposition, key)
        elif hasattr(result, key):
            actual = getattr(result, key)
        else:
            continue
        passed = abs(float(actual) - float(value)) < 1e-10
        logger.log_case(filename, value, actual, passed, f"hand-authored expected {key}")
        assert passed, (key, actual, value)

    assert result.edges == [float(value) for value in reference["edges"]]
    logs = read_jsonl(log_path)
    assert logs[0]["run_id"].startswith("pytest-" + fixture["name"])
    assert logs[0]["input_digest"] == filename
    assert all(item["status"] in {"success", "failed"} for item in logs)


def test_perfect_calibration_exact_hand_values():
    result = analyze(load_fixture("perfect_calibration.json"))

    assert result.decomposition.reliability == 0.0
    assert result.decomposition.raw_brier == pytest.approx(Fraction(7, 32), abs=0)
    assert result.weights.sum_weight_fractions == pytest.approx(1.0, abs=0)


def test_extreme_weights_ess_and_negative_residual_explicit():
    result = analyze(load_fixture("extreme_weights.json"))

    assert result.weights.total_weight == 201.0
    assert result.weights.effective_sample_size == pytest.approx(40401 / 20001, rel=1e-15)
    assert result.decomposition.decomposition_residual < 0
    assert result.bins[1].zero_weight_count == 1


def test_empty_fixed_bins_remain_separate_and_weight_conserved():
    result = analyze(load_fixture("empty_fixed_bins.json"))

    assert len(result.bins) == 10
    assert [item.empty for item in result.bins].count(True) == 9
    assert sum(item.weight_fraction for item in result.bins) == pytest.approx(1.0, abs=0)
    assert [item for item in result.bins if not item.empty][0].observed_rate == pytest.approx(2 / 3)
