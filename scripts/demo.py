"""Local end-to-end demonstration using the service layer and SQLite."""

from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lag_correlation.contracts import AnalysisRequest, LagGrid, MatchingWindow, SeriesPayload
from lag_correlation.repository import AnalysisRepository
from lag_correlation.service import result_to_response, run_analysis


def main() -> int:
    database_path = os.path.join(PROJECT_ROOT, "data", "demo.sqlite3")
    if os.path.exists(database_path):
        os.remove(database_path)
    repository = AnalysisRepository(database_path)

    times = [float(index * 10) for index in range(8)]
    values = [1.0, 4.0, 2.0, 8.0, 3.0, 7.0, 2.0, 5.0]
    dense_times: list[float] = []
    dense_values: list[float] = []
    for time_value, value in zip(times, values):
        for offset in (-0.15, 0.0, 0.15):
            dense_times.append(time_value - 3.0 + offset)
            dense_values.append(value + offset)

    request = AnalysisRequest(
        request_id="demo-known-lag-3",
        x=SeriesPayload(name="sparse_x", times=times, values=values),
        y=SeriesPayload(name="dense_y", times=dense_times, values=dense_values),
        lag_grid=LagGrid(min_lag=-6.0, max_lag=6.0, step=1.0),
        matching=MatchingWindow(tolerance=0.2),
    )

    result = run_analysis(request)
    repository.save(result)
    stored = repository.get_response_dict(result.request_id)

    print(
        json.dumps(
            {
                "stored_request_retrieved": stored["request_id"] == result.request_id,
                "response_summary": {
                    "request_id": stored["request_id"],
                    "status": stored["status"],
                    "failure_category": stored["failure_category"],
                    "peak_lag": stored["peak_lag"],
                    "peak_correlation": stored["peak_correlation"],
                    "anchor_side": stored["anchor_side"],
                    "n_pairs": stored["best_lag"]["n_pairs"],
                    "dense_samples_used": stored["best_lag"]["dense_samples_used"],
                    "dense_samples_unmatched": stored["best_lag"]["dense_samples_unmatched"],
                    "schema_version": stored["schema_version"],
                    "algorithm_version": stored["algorithm_version"],
                },
                "causality_notice": stored["causality_notice"],
                "database_path": database_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
