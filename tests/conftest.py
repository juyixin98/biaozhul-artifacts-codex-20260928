from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(304)


@pytest.fixture
def x_data() -> np.ndarray:
    return np.linspace(0.0, 2.0, 14)


@pytest.fixture
def noisy(rng) -> np.ndarray:
    return rng.normal(0.0, 5e-4, size=14)


@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    return tmp_path / "runs.jsonl"


@pytest.fixture
def read_jsonl():
    def _read(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    return _read
