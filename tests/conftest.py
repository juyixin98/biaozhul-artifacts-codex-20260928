from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcmc_diagnostics.config import Settings
from mcmc_diagnostics.main import create_app


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        database_path=tmp_path / "test-diagnostics.db",
        default_max_acf_lag=20,
        max_chains=10,
        max_draws_per_chain=10_000,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client
