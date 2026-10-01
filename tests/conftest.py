"""Shared test fixtures."""

from __future__ import annotations

import pytest

from app.parsing import EXACT_POLICY
from app.core import create_store


@pytest.fixture
def empty_store():
    return create_store(
        problem_id="unit-problem",
        directions=["min", "max"],
        policy=EXACT_POLICY,
    )
