from __future__ import annotations

import sqlite3

import pytest

from app.errors import PersistenceError, ResourceExhaustedError
from app.persistence.repository import SQLiteRepository


def test_sqlite_full_disk_is_resource_exhausted(service, monkeypatch, test_logger):
    original_connect = sqlite3.connect

    class FullDiskConnection:
        row_factory = None

        def __init__(self, *args, **kwargs):
            pass

        def execute(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("disk I/O error: no space left on device")

        def executescript(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("disk I/O error: no space left on device")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def close(self):
            pass

    exhausted_repository = SQLiteRepository(service.repository.path)
    exhausted_repository.connection_factory = FullDiskConnection
    service.repository = exhausted_repository
    with pytest.raises(ResourceExhaustedError) as exc_info:
        service.create_model(1.0, 1.0)

    test_logger.note(
        "sqlite_full_disk",
        "error_classification",
        {
            "category": exc_info.value.category,
            "code": exc_info.value.code,
            "sqlite_error": exc_info.value.details["sqlite_error"],
        },
        "A full local SQLite media is classified separately from a generic persistence failure.",
    )
    assert exc_info.value.category == "resource_exhausted"


def test_generic_sqlite_error_is_persistence_failure(service, monkeypatch):
    def broken_connect(*_args, **_kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    service.repository.connection_factory = broken_connect
    with pytest.raises(PersistenceError) as exc_info:
        service.repository._connect()
    assert exc_info.value.category == "persistence_failed"
