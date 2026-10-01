#!/usr/bin/env python3
"""Run a synthetic request against the local FastAPI application."""

from __future__ import annotations

from datetime import datetime
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient

from lagcorr.api import create_app


def main() -> int:
    app = create_app()
    with TestClient(app) as client:
        times = [float(i) for i in range(24)]
        payload = {
            "request_id": f"demo-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "a": {
                "times": times,
                "values": [math.sin(i / 4.0) for i in range(24)],
            },
            "b": {
                "times": [time + 2.0 for time in times],
                "values": [math.sin(i / 4.0) for i in range(24)],
            },
            "lags": [2.0, 20.0],
            "tolerance": 0.4,
            "include_pairs": True,
        }
        response = client.post("/analyses", json=payload)

    print(f"POST /analyses -> HTTP {response.status_code}")
    if response.status_code != 201:
        print(response.text)
        return 1

    body = response.json()
    print(f"request_id={body['request_id']} version={body['result_version']}")
    print(
        f"peak_lag={body['peak_lag']} "
        f"peak_correlation={body['peak_correlation']:.12f} "
        f"peak_kind={body['peak_kind']}"
    )
    for result in body["results"]:
        correlation = (
            "n/a" if result["correlation"] is None else f"{result['correlation']:.12f}"
        )
        diagnosis = result["diagnosis"]
        print(
            "  "
            f"lag={result['lag']} status={result['status']} r={correlation} "
            f"candidate_edges={diagnosis['candidate_edges']} "
            f"pairs={result['pair_count']} fraction={diagnosis['pair_fraction']:.2f} "
            f"reason={result['reason_code']}"
        )
    print(body["warning"])
    print("Redacted diagnostics printed: raw sample values were not logged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
