"""SQLite persistence for decomposition runs and replayable diagnostics."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('running', 'complete', 'failed')),
    observations INTEGER NOT NULL,
    request_fingerprint TEXT NOT NULL,
    kernel_version TEXT,
    schema_version TEXT,
    config_json TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT,
    error_category TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS run_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    run_number INTEGER,
    iteration INTEGER,
    stage TEXT NOT NULL,
    judgment TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_run_events_run_id ON run_events(run_id, id);
"""


class RunRepository:
    def __init__(self, path: str) -> None:
        self.path = path
        # FastAPI dependencies may execute through a worker thread while the
        # TestClient builds the dependency callable on another. Each request
        # gets a fresh connection, so cross-thread sharing is short-lived.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RunRepository":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()

    def create_run(
        self,
        *,
        run_id: str,
        observations: int,
        request_fingerprint: str,
        config: dict[str, Any],
        request_payload: dict[str, Any],
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO runs (
                run_id, status, observations, request_fingerprint,
                config_json, request_json
            ) VALUES (?, 'running', ?, ?, ?, ?)
            """,
            (
                run_id,
                observations,
                request_fingerprint,
                json.dumps(config, sort_keys=True, separators=(",", ":")),
                json.dumps(request_payload, sort_keys=True, separators=(",", ":")),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def add_event(
        self,
        *,
        run_id: str,
        run_number: int | None,
        stage: str,
        judgment: str,
        state: dict[str, Any],
        iteration: int | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO run_events
                (run_id, run_number, iteration, stage, judgment, state_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                run_number,
                iteration,
                stage,
                judgment,
                json.dumps(state, sort_keys=True, allow_nan=False),
            ),
        )

    def complete_run(
        self,
        *,
        run_number: int,
        kernel_version: str,
        schema_version: str,
        result: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """
            UPDATE runs
               SET status = 'complete',
                   kernel_version = ?,
                   schema_version = ?,
                   result_json = ?,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = ?
            """,
            (
                kernel_version,
                schema_version,
                json.dumps(result, sort_keys=True, allow_nan=False),
                run_number,
            ),
        )
        self.conn.commit()

    def fail_run(
        self,
        *,
        run_number: int,
        category: str,
        code: str,
        message: str,
    ) -> None:
        self.conn.execute(
            """
            UPDATE runs
               SET status = 'failed',
                   error_category = ?,
                   error_code = ?,
                   error_message = ?,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = ?
            """,
            (category, code, message, run_number),
        )
        self.conn.commit()

    def list_events(self, run_id: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM run_events WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        )
