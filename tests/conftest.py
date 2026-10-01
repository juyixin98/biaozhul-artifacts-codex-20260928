from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from survey_backend.api import create_app
from survey_backend.config import load_settings


@pytest.fixture
def client():
    settings = load_settings("config/test.env")
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def settings():
    return load_settings("config/test.env")
