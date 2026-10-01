from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def client(tmp_path):
    settings = Settings(db_path=str(tmp_path / "test-survival.sqlite3"))
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def analysis_payload():
    return {
        "client_request_id": "client-hand-001",
        "analysis_name": "hand fixture",
        "confidence_level": 0.95,
        "significance_level": 0.05,
        "confidence_interval_method": "greenwood_linear",
        "groups": ["A", "B", "EMPTY"],
        "observations": [
            {"observation_id": "A1", "group": "A", "time": 2.0, "event": True, "entry": 0.0},
            {"observation_id": "A2", "group": "A", "time": 4.0, "event": False, "entry": 0.0},
            {"observation_id": "A3", "group": "A", "time": 5.0, "event": True, "entry": 0.0},
            {"observation_id": "A4", "group": "A", "time": 6.0, "event": False, "entry": 0.0},
            {"observation_id": "A5", "group": "A", "time": 8.0, "event": False, "entry": 0.0},
            {"observation_id": "B1", "group": "B", "time": 3.0, "event": True, "entry": 0.0},
            {"observation_id": "B2", "group": "B", "time": 7.0, "event": True, "entry": 0.0},
        ],
    }
