from __future__ import annotations

import json

from experiments.run_experiment import run


def test_reproducible_experiment_report(tmp_path) -> None:
    report = run(tmp_path)
    assert report["all_passed"] is True
    saved = json.loads((tmp_path / "experiment_report.json").read_text())
    assert {check["name"] for check in saved["checks"]} == {
        "split_equals_combined_posterior",
        "prediction_matches_scipy_betabinom",
        "revision_withdraws_old_sufficient_statistics",
        "revision_creates_versioned_result",
        "large_counts_remain_finite",
    }
    assert "versions" in saved["environment"]
