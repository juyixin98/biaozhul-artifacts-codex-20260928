"""SQLite persistence for versioned results and run-scoped event logs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .errors import StorageError
from .models import utc_now_iso


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    input_digest TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT,
    error_category TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_segments (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    start_index INTEGER NOT NULL,
    end_index INTEGER NOT NULL,
    length INTEGER NOT NULL,
    mean REAL NOT NULL,
    cost REAL NOT NULL,
    penalty REAL NOT NULL,
    PRIMARY KEY (run_id, seq),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS run_events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    level TEXT NOT NULL CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR')),
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, seq),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_digest ON runs(input_digest);
"""


class SQLiteRepository:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.db_path,
                timeout=10.0,
                isolation_level=None,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA journal_mode=WAL")
            return connection
        except sqlite3.Error as exc:
            raise StorageError(f"cannot open database {self.db_path}: {exc}") from exc

    def initialize(self) -> None:
        try:
            with self.connect() as connection:
                connection.executescript(SCHEMA)
        except sqlite3.Error as exc:
            raise StorageError(f"database migration failed: {exc}") from exc

    def create_run(
        self,
        *,
        run_id: str,
        input_digest: str,
        algorithm: str,
        request_payload: dict[str, Any],
    ) -> str:
        created_at = utc_now_iso()
        try:
            with self.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, status, input_digest, algorithm,
                        request_json, created_at, updated_at
                    ) VALUES (?, 'running', ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        input_digest,
                        algorithm,
                        json.dumps(request_payload, allow_nan=False),
                        created_at,
                        created_at,
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"failed to create run {run_id}: {exc}") from exc
        return created_at

    def append_event(
        self,
        *,
        run_id: str,
        level: str,
        event_type: str,
        message: str,
        details: dict[str, Any],
    ) -> None:
        try:
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq "
                    "FROM run_events WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                connection.execute(
                    """
                    INSERT INTO run_events (
                        run_id, seq, level, event_type, message,
                        details_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        int(row["next_seq"]),
                        level,
                        event_type,
                        message,
                        json.dumps(details, allow_nan=False, sort_keys=True),
                        utc_now_iso(),
                    ),
                )
                connection.execute("COMMIT")
        except sqlite3.Error as exc:
            raise StorageError(f"failed to append event for {run_id}: {exc}") from exc

    def mark_failed(self, *, run_id: str, category: str, message: str) -> None:
        try:
            with self.connect() as connection:
                connection.execute(
                    """
                    UPDATE runs
                    SET status='failed', error_category=?, error_message=?,
                        updated_at=?
                    WHERE run_id=?
                    """,
                    (category, message, utc_now_iso(), run_id),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"failed to mark run {run_id} failed: {exc}") from exc

    def save_result(
        self,
        *,
        run_id: str,
        result: dict[str, Any],
        segments: list[dict[str, Any]],
    ) -> None:
        try:
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    UPDATE runs
                    SET status='success', result_json=?, error_category=NULL,
                        error_message=NULL, updated_at=?
                    WHERE run_id=?
                    """,
                    (
                        json.dumps(result, allow_nan=False, sort_keys=True),
                        utc_now_iso(),
                        run_id,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO run_segments (
                        run_id, seq, start_index, end_index, length, mean,
                        cost, penalty
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            index,
                            segment["start"],
                            segment["end"],
                            segment["length"],
                            segment["mean"],
                            segment["cost"],
                            segment["penalty"],
                        )
                        for index, segment in enumerate(segments)
                    ],
                )
                connection.execute("COMMIT")
        except sqlite3.Error as exc:
            raise StorageError(f"failed to persist result {run_id}: {exc}") from exc

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()

    def get_events(self, run_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM run_events WHERE run_id=? ORDER BY seq",
                    (run_id,),
                ).fetchall()
            )
