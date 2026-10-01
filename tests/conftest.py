from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_local_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("MCMC_DIAG_ROOT", str(tmp_path))
    monkeypatch.setenv("MCMC_DIAG_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("MCMC_DIAG_LOG", str(tmp_path / "test.log"))
    from app.mcmc_diagnostics.dependencies import cached_repository, cached_settings

    cached_settings.cache_clear()
    cached_repository.cache_clear()
    yield
    cached_settings.cache_clear()
    cached_repository.cache_clear()


@pytest.fixture
def ar1_data():
    import numpy as np

    rng = np.random.default_rng(20260930)
    n_chains, n_draws, phi = 4, 5_000, 0.8
    data = np.empty((n_chains, n_draws))
    for chain in range(n_chains):
        eps = rng.normal(size=n_draws)
        data[chain, 0] = eps[0]
        for t in range(1, n_draws):
            data[chain, t] = phi * data[chain, t - 1] + eps[t]
    return data


@pytest.fixture
def shifted_data():
    import numpy as np

    rng = np.random.default_rng(20260931)
    shifts = np.array([-3.0, -1.0, 1.0, 3.0])
    return rng.normal(size=(4, 2_000)) + shifts[:, None]
