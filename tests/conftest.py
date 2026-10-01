from __future__ import annotations

import json
import pathlib

import pytest


FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def robust_fixture() -> dict:
    with (FIXTURE_DIR / "robust_stl.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)
