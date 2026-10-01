from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    path = tmp_path / "test-logs"
    path.mkdir()
    return path
