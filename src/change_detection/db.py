from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .errors import PersistenceError, RunNotFoundError

SCHEMA_SQL = """
PRAGMA user_version = 1;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    core_version TEXT NOT NULL,
    status TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT,
    error_category TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    event TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_run_events_run_id ON run_events(run_id, id);
"""


class RunRepository:
    def __init__(self, database_path: str):
        self.database_path = database_path
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        try:
            with self.connect() as connection:
                connection.executescript(SCHEMA_SQL)
        except sqlite3.Error as exc:
            raise PersistenceError(f"failed to initialize SQLite database: {exc}") from exc

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def create_run(
        self,
        *,
        run_id: str,
        schema_version: int,
        core_version: str,
        input_sha256: str,
        request: dict[str, Any],
        created_at: str,
    ) -> None:
        try:
            with self.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, schema_version, core_version, status, input_sha256,
                        request_json, result_json, error_category, error_message,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
                    """,
                    (
                        run_id,
                        schema_version,
                        core_version,
                        "RUNNING",
                        input_sha256,
                        self._json(request),
                        created_at,
                        created_at,
                    ),
                )
        except sqlite3.Error as exc:
            raise PersistenceError(f"failed to create run {run_id}: {exc}") from exc

    def append_event(self, run_id: str, event: str, payload: dict[str, Any], created_at: str) -> None:
        try:
            with self.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO run_events (run_id, event, payload_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, event, self._json(payload), created_at),
                )
        except sqlite3.Error as exc:
            raise PersistenceError(f"failed to append event for run {run_id}: {exc}") from exc

    def complete_run(self, *, run_id: str, result: dict[str, Any], updated_at: str) -> None:
        try:
            with self.connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE runs
                    SET status = ?, result_json = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    ("SUCCEEDED", self._json(result), updated_at, run_id),
                )
                if cursor.rowcount != 1:
                    raise RunNotFoundError(f"run {run_id} does not exist")
        except sqlite3.Error as exc:
            raise PersistenceError(f"failed to complete run {run_id}: {exc}") from exc

    def fail_run(self, *, run_id: str, category: str, message: str, updated_at: str) -> None:
        try:
            with self.connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE runs
                    SET status = ?, error_category = ?, error_message = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    ("FAILED", category, message, updated_at, run_id),
                )
                if cursor.rowcount != 1:
                    raise RunNotFoundError(f"run {run_id} does not exist")
        except sqlite3.Error as exc:
            raise PersistenceError(f"failed to mark run {run_id} as failed: {exc}") from exc

    def get_run(self, run_id: str) -> dict[str, Any]:
        try:
            with self.connect() as connection:
                row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
                if row is None:
                    raise RunNotFoundError(f"run {run_id} does not exist")
                event_rows = connection.execute(
                    """
                    SELECT event, payload_json, created_at
                    FROM run_events
                    WHERE run_id = ?
                    ORDER BY id
                    """,
                    (run_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise PersistenceError(f"failed to read run {run_id}: {exc}") from exc

        return {
            "run_id": row["run_id"],
            "schema_version": row["schema_version"],
            "core_version": row["core_version"],
            "status": row["status"],
            "input_sha256": row["input_sha256"],
            "request": json.loads(row["request_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error_category": row["error_category"],
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "events": [
                {
                    "event": event_row["event"],
                    "payload": json.loads(event_row["payload_json"]),
                    "created_at": event_row["created_at"],
                }
                for event_row in event_rows
            ],
        }

    def health(self) -> bool:
        try:
            with self.connect() as connection:
                connection.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False
