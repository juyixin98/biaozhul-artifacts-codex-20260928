"""SQLite-backed result repository."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import Settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS diagnostic_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE,
    endpoint TEXT NOT NULL,
    status TEXT NOT NULL,
    n_chains INTEGER NOT NULL,
    total_draws INTEGER NOT NULL,
    schema_version TEXT NOT NULL,
    kernel_version TEXT NOT NULL,
    algorithm_revision TEXT NOT NULL,
    result_json TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_diagnostic_runs_created ON diagnostic_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_diagnostic_runs_status ON diagnostic_runs(status);
"""


class DiagnosticRepository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.settings.database_path, timeout=10, check_same_thread=False
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def save_success(self, endpoint: str, result: dict[str, Any]) -> int:
        versions = result.get("versions", {})
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO diagnostic_runs (
                    request_id, endpoint, status, n_chains, total_draws,
                    schema_version, kernel_version, algorithm_revision, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.get("request_id"),
                    endpoint,
                    result.get("status"),
                    result.get("n_chains"),
                    result.get("total_draws"),
                    versions.get("schema_version"),
                    versions.get("kernel_version"),
                    versions.get("algorithm_revision"),
                    json.dumps(result, ensure_ascii=True, sort_keys=True),
                ),
            )
            return int(cursor.lastrowid)

    def save_failure(
        self,
        endpoint: str,
        request_id: str,
        error_code: str,
        error_message: str,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO diagnostic_runs (
                    request_id, endpoint, status, n_chains, total_draws,
                    schema_version, kernel_version, algorithm_revision,
                    error_code, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    endpoint,
                    "error",
                    0,
                    0,
                    self.settings.schema_version,
                    self.settings.kernel_version,
                    self.settings.algorithm_revision,
                    error_code,
                    error_message,
                ),
            )
            return int(cursor.lastrowid)

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM diagnostic_runs WHERE request_id = ?", (request_id,)
            ).fetchone()
        return dict(row) if row else None
