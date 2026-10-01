from __future__ import annotations

import json

from survey_imputation.experiment import run_comparison, write_evidence
from survey_imputation.repository import SQLiteResultRepository


def test_reproducible_comparison_writes_evidence_and_sqlite(tmp_path):
    repository = SQLiteResultRepository(tmp_path / "experiment.db")

    first = run_comparison(repository=repository, n=200, missing_probability=0.2, m=8, base_seed=1729)
    second = run_comparison(repository=None, n=200, missing_probability=0.2, m=8, base_seed=1729)

    assert [row.estimate for row in first] == [row.estimate for row in second]
    assert len(first) == 10

    mechanisms = [row.mechanism for row in first]
    assert mechanisms.count("complete") == 1
    assert mechanisms.count("mcar") == 3
    assert mechanisms.count("mar") == 3
    assert mechanisms.count("mnar") == 3

    by_method = {(row.mechanism, row.method): row for row in first}
    mcar_target = by_method[("mcar", "complete_case")].target_value
    assert by_method[("mcar", "complete_case")].estimate == by_method[("mcar", "mean_imputation")].estimate
    assert (
        by_method[("mcar", "mean_imputation")].standard_error
        < by_method[("mcar", "complete_case")].standard_error
    )
    assert by_method[("mcar", "multiple_imputation")].imputation_count == 8
    mar_target = by_method[("mar", "mean_imputation")].target_value
    assert abs(by_method[("mar", "multiple_imputation")].estimate - mar_target) < abs(
        by_method[("mar", "mean_imputation")].estimate - mar_target
    )

    output = write_evidence(first, tmp_path / "results" / "evidence.json")
    payload = json.loads(output.read_text())
    assert len(payload["rows"]) == 10
    assert any("MNAR" in assumption and "cannot guarantee" in assumption for assumption in payload["assumptions"])

    stored_rows = repository.list_experiment_rows()
    assert len(stored_rows) == 10
