from __future__ import annotations

import json
import math

import pytest

from experiments.run_experiment import run


def test_reproduction_experiment_uses_independent_expected_judgement():
    report = run()
    by_name = {scenario["name"]: scenario for scenario in report["scenarios"]}

    small = by_name["hand_small_counts"]
    assert small["posterior"]["alpha"] == 3.0
    assert small["posterior"]["beta"] == 2.0
    assert small["predictive"]["first_five"][:3] == pytest.approx(
        [0.2, 0.4, 0.4], abs=1e-12
    )

    all_success = by_name["all_success"]
    assert all_success["posterior"]["alpha"] == 10001.0
    assert all_success["posterior"]["beta"] == 1.0

    for scenario in report["scenarios"]:
        prediction = scenario["predictive"]
        assert prediction["all_finite"] is True
        assert prediction["nonnegative"] is True
        assert math.isclose(prediction["probability_sum"], 1.0, abs_tol=1e-10)
        assert "sum to one" in scenario["judgement"]

    serialized = json.dumps(report, allow_nan=False)
    assert "NaN" not in serialized
