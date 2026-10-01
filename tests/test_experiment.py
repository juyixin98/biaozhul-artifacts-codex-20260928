from __future__ import annotations

import json

from experiments.reproduce import run_experiment, write_markdown


def test_experiment_summary_contains_known_targets_and_methods(tmp_path):
    summary = run_experiment(repetitions=2, n=120, seed=909, m=2, burn_in=1)
    assert summary["config"]["repetitions"] == 2
    assert summary["config"]["target_coefficient_order"] == [
        "beta_q1",
        "beta_q2",
        "intercept",
    ]
    for mechanism in ("mcar", "mar", "mnar"):
        methods = summary["mechanisms"][mechanism]
        assert set(methods) == {
            "baseline",
            "complete_case",
            "mean_imputation",
            "multiple_imputation",
        }
        for method, values in methods.items():
            assert len(values["bias"]) == 3
            assert len(values["coverage_95"]) == 3
            assert all(0.0 <= coverage <= 1.0 for coverage in values["coverage_95"])

    output = tmp_path / "summary.json"
    output.write_text(json.dumps(summary), encoding="utf-8")
    write_markdown(summary, output.with_suffix(".md"))
    assert "MNAR" in output.with_suffix(".md").read_text(encoding="utf-8")
