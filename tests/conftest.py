from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.fixture
def two_worker_problem():
    rng = np.random.default_rng(310)
    # Different sample counts create asymmetric local curvature.
    A1 = rng.normal(size=(5, 3))
    A2 = rng.normal(size=(13, 3))
    b1 = rng.normal(size=5)
    b2 = rng.normal(size=13)
    # f_i(x) = 0.5 ||A_i x - b_i||^2
    Q1, q1 = A1.T @ A1, -A1.T @ b1
    Q2, q2 = A2.T @ A2, -A2.T @ b2
    yield {"Q": (Q1, Q2), "q": (q1, q2), "A": (A1, A2), "b": (b1, b2)}


@pytest.fixture
def isolated_app(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMM_DATA_DIR", str(tmp_path / "runs-data"))
    monkeypatch.setenv("ADMM_MAX_DIMENSION", "8")
    monkeypatch.setenv("ADMM_MAX_ITERATIONS", "80")
    monkeypatch.setenv("ADMM_MAX_RUNS", "3")
    import admm_consensus.app as app_module

    importlib.reload(app_module)
    app_module.settings = app_module.Settings.from_env()
    app_module.repository = app_module.RunRepository(app_module.settings)
    app_module.coordinator = app_module.Coordinator(app_module.repository)
    app_module.app.state.settings = app_module.settings
    app_module.app.state.repository = app_module.repository
    app_module.app.state.coordinator = app_module.coordinator
    return app_module
