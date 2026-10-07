from __future__ import annotations

import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from ddmin_service.api import create_app
from ddmin_service.config import Settings

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROGRAMS = ROOT / "fixtures" / "programs"
INPUTS = ROOT / "fixtures" / "inputs"


@pytest.fixture
def settings(tmp_path):
    return Settings(db_path=str(tmp_path / "test.db"), default_timeout_seconds=2.0)


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def load_input(name: str) -> dict:
    return json.loads((INPUTS / name).read_text())


def make_spec(program: str, input_name: str, timeout: float = 2.0) -> dict:
    return {
        "program": ["python3", str(PROGRAMS / program)],
        "input": load_input(input_name),
        "classifier": {
            "target_exit_codes": [3],
            "pass_exit_codes": [0],
            "timeout_seconds": timeout,
        },
        "sensitive_fields": ["secret"],
    }
