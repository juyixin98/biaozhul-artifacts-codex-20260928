from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text("utf-8"))


@pytest.fixture(scope="session")
def primary_fixture() -> dict[str, Any]:
    return load_fixture("synthetic_stl_70.json")


@pytest.fixture(scope="session")
def boundary_fixture() -> dict[str, Any]:
    return load_fixture("minimum_periods_14.json")

