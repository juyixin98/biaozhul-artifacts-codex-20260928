#!/usr/bin/env python3
"""Run a self-contained local API demonstration without external services."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PERM_DB_PATH", str(ROOT / "data" / "demo.sqlite"))
os.environ.setdefault("PERM_EXACT_MAX_COMBINATIONS", "100000")
os.environ.setdefault("PERM_DEFAULT_MONTE_CARLO_REPLICATIONS", "999")

from fastapi.testclient import TestClient  # noqa: E402

from permutation_service.api import create_app  # noqa: E402
from permutation_service.config import Settings  # noqa: E402


def wait_for_completion(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/runs/{run_id}")
        if response.status_code in {200, 400, 409, 500}:
            return response.json()
        time.sleep(0.01)
    raise RuntimeError(f"run {run_id} did not finish")


def print_run(title: str, run: dict) -> None:
    print(f"\n== {title} ==")
    print(f"run_id: {run['run_id']}")
    print(f"status: {run['status']}")
    print(f"input_hash: {run['input_hash']}")
    if run["result"]:
        result = run["result"]
        print(f"method: {result['method_used']}")
        print(
            "observed statistic: "
            f"{result['statistic']:.12g}"
        )
        print(f"two-sided p value: {result['p_value']:.12g}")
        print(
            "valid permutations: "
            f"{result['evaluated_permutations']} / "
            f"{result['total_possible_assignments']}"
        )
        if result["standard_error"] is not None:
            print(f"standard error: {result['standard_error']:.12g}")
            low, high = result["confidence_interval_95"]
            print(f"95% normal interval: [{low:.12g}, {high:.12g}]")
        print(f"decision at alpha={result['alpha']}: {result['significant']}")
    if run["error"]:
        print(f"error: {run['error']['code']} - {run['error']['message']}")


def main() -> None:
    settings = Settings.from_env()
    app = create_app(settings)
    with TestClient(app) as client:
        print("== versions ==")
        print(client.get("/version").json())

        exact = client.post(
            "/api/v1/runs",
            json={"values": [1.0, 2.0, 3.0], "group1_size": 1},
        ).json()
        exact_run = wait_for_completion(client, exact["run_id"])
        print_run("Exact hand-checkable run", exact_run)

        monte_carlo = client.post(
            "/api/v1/runs",
            json={
                "values": list(range(12)),
                "group1_size": 6,
                "method": "auto",
                "exact_max_combinations": 0,
                "monte_carlo_replications": 500,
                "random_seed": 20260930,
            },
        ).json()
        mc_run = wait_for_completion(client, monte_carlo["run_id"])
        print_run("Seeded Monte Carlo run", mc_run)

        invalid = client.post(
            "/api/v1/runs",
            json={"values": [1.0, 2.0], "group1_size": 2},
        )
        print("\n== Synchronous input error ==")
        print(f"HTTP {invalid.status_code}")
        print(invalid.json()["error"])

        events = client.get(
            f"/api/v1/runs/{exact_run['run_id']}/events"
        ).json()
        print("\n== Exact run audit trail ==")
        for event in events:
            print(f"{event['seq']:>2} {event['phase']}: {event['message']}")


if __name__ == "__main__":
    main()

