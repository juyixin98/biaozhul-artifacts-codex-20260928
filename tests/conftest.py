from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CHG_DATABASE_PATH", str(ROOT / "data" / "test.db"))
os.environ.setdefault("CHG_LOG_PATH", str(ROOT / "logs" / "test.log"))
os.environ.setdefault("CHG_LOG_LEVEL", "DEBUG")


@pytest.fixture
def run_id_context(caplog):
    caplog.set_level(logging.DEBUG, logger="change_detection")
    return "pytest-run-identity"

