from __future__ import annotations

from pathlib import Path

import pytest

from milp.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        journal_dir=tmp_path / "logs",
        default_integer_tolerance=1e-8,
        default_constraint_tolerance=1e-8,
        default_lp_tolerance=1e-10,
        default_time_limit_seconds=5.0,
        default_node_limit=100,
        max_events_in_response=100,
    )
