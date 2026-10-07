from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.models import Document
from app.reducer import Reducer
from app.store import Store

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET = str(BASE_DIR / "fixtures" / "target_program.py")
SAMPLE = BASE_DIR / "fixtures" / "sample_input.txt"


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        db_path=str(tmp_path / "test.db"),
        target_program=TARGET,
        run_timeout_sec=1.0,
    )


@pytest.fixture()
def store(settings):
    s = Store(settings.db_path)
    yield s
    s.close()


@pytest.fixture()
def reducer(settings, store) -> Reducer:
    return Reducer(settings, store)


@pytest.fixture()
def sample_doc() -> Document:
    return Document.from_text(SAMPLE.read_text(encoding="utf-8"))
