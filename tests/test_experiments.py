from __future__ import annotations

import json

from scripts.run_experiments import run


def test_reproducible_report_contains_concrete_oracle_checks(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "report.json"
    monkeypatch.setattr("scripts.run_experiments.REPORT", report_path)
    report = run()

    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded == report

    high_dim = report["fixtures"]["high_dim_small_sample"]
    assert high_dim["shape"] == [14, 22]
    assert high_dim["robust_tyler"]["rank"] == 22
    assert high_dim["robust_tyler"]["invertible"] is True
    assert abs(
        high_dim["robust_tyler"]["standardized_min_eigenvalue"]
        - high_dim["statsmodels_standardized_reference"]["min_eigenvalue"]
    ) < 2e-9
    assert abs(
        high_dim["robust_tyler"]["standardized_condition_number"]
        - high_dim["statsmodels_standardized_reference"]["condition_number"]
    ) < 2e-7

    outliers = report["fixtures"]["strong_outliers"]
    assert abs(
        outliers["robust_tyler"]["first_weight"]
        - outliers["statsmodels_standardized_reference"]["first_weight"]
    ) < 1e-8

    lw = report["fixtures"]["ledoit_wolf_collinear"]
    assert lw["production"]["rank"] == 6
    assert (
        lw["production"]["standardized_min_eigenvalue"]
        == lw["independent_formula_reference"]["min_eigenvalue"]
    )
    assert (
        lw["production"]["standardized_condition_number"]
        == lw["independent_formula_reference"]["condition_number"]
    )
