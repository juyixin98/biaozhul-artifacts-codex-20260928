from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PROJECT = ROOT / "fixtures" / "sample_algorithm_pkg"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def fixture_project() -> Path:
    return FIXTURE_PROJECT


@pytest.fixture
def test_command() -> list[str]:
    return ["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
