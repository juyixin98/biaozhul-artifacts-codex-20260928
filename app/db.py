"""SQLite-backed request audit log."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .logging_setup import get_logger


SCHEMA = """
CREATE TABLE IF NOT EXISTS request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    error_category TEXT,
    message TEXT,
    processing_location TEXT NOT NULL,
    estimator_versions TEXT,
    processing_steps TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_request_log_request_id
    ON request_log(request_id);
"""


def connect(sqlite_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(sqlite_path, timeout=5, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(sqlite_path: Path) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(sqlite_path) as connection:
        connection.executescript(SCHEMA)


def insert_request_log(
    *,
    sqlite_path: Path,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    processing_location: str,
    estimator_versions: Sequence[str] | None = None,
    processing_steps: Sequence[dict[str, Any]] | None = None,
    error_category: str | None = None,
    message: str | None = None,
) -> None:
    try:
        initialize_database(sqlite_path)
        with connect(sqlite_path) as connection:
            connection.execute(
                """
                INSERT INTO request_log (
                    request_id, method, path, status_code, error_category, message,
                    processing_location, estimator_versions, processing_steps
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    method,
                    path,
                    status_code,
                    error_category,
                    message,
                    processing_location,
                    json.dumps(list(estimator_versions or []), ensure_ascii=False),
                    json.dumps(list(processing_steps or []), ensure_ascii=False),
                ),
            )
    except Exception:
        # Audit failure must not erase the statistical response.
        get_logger().exception("failed to write SQLite request audit row")
