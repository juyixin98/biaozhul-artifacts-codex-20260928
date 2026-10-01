from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def temp_db(monkeypatch, tmp_path: Path) -> Path:
    db_path = tmp_path / "test-requests.sqlite3"
    monkeypatch.setenv("COVARIANCE_DB", str(db_path))
    import app.config as config

    monkeypatch.setattr(config.settings, "database_path", db_path)
    import app.persistence as persistence

    monkeypatch.setattr(persistence, "settings", config.settings)
    import app.main as main

    monkeypatch.setattr(main, "ensure_database", persistence.ensure_database)
    monkeypatch.setattr(main, "record_request", persistence.record_request)
    return db_path


@pytest.fixture()
def client(temp_db: Path) -> TestClient:
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
