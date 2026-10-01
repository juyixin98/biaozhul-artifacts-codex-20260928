from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from permutation_service.config import Settings
from permutation_service.api import create_app


@pytest.fixture()
def settings(tmp_path):
    return Settings(
        db_path=tmp_path / "test-runs.sqlite",
        default_monte_carlo_replications=99,
        exact_max_combinations=100,
        requested_exact_limit_cap=50_000_000,
        max_values=1000,
        max_monte_carlo_replications=10_000,
    )


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def wait_for_run(client, run_id: str, *, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/runs/{run_id}")
        last = response
        if response.status_code in {200, 400, 409, 500}:
            return response.json()
        time.sleep(0.01)
    raise AssertionError(f"run did not finish; last response={last}")

