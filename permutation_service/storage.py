"""SQLite persistence for run status, results, failures, and progress events."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class RunStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.db_path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            if self.db_path != ":memory:":
                connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA busy_timeout = 30000")
            self._local.connection = connection
        return connection

    def init_db(self) -> None:
        with self._write_lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'running', 'completed', 'failed', 'canceled')
                    ),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    seq INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    completed INTEGER,
                    total INTEGER,
                    batch_size INTEGER,
                    message TEXT NOT NULL,
                    UNIQUE(run_id, seq)
                );

                CREATE INDEX IF NOT EXISTS idx_run_events_run_seq
                    ON run_events(run_id, seq);
                """
            )

    @staticmethod
    def _dumps(payload: Any | None) -> str | None:
        return None if payload is None else json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _loads(raw: str | None) -> Any | None:
        return None if raw is None else json.loads(raw)

    def _row_to_run(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "input": self._loads(row["input_json"]),
            "input_hash": row["input_hash"],
            "result": self._loads(row["result_json"]),
            "error": self._loads(row["error_json"]),
            "cancel_requested": bool(row["cancel_requested"]),
        }

    def create_run(self, run_input: dict[str, Any], input_hash: str) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        now = utc_now()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, status, created_at, updated_at,
                    input_hash, input_json, cancel_requested
                ) VALUES (?, 'pending', ?, ?, ?, ?, 0)
                """,
                (run_id, now, now, input_hash, self._dumps(run_input)),
            )
            connection.execute(
                """
                INSERT INTO run_events(
                    run_id, seq, created_at, phase, completed, total,
                    batch_size, message
                ) VALUES (?, 0, ?, 'accepted', 0, 0, 0, ?)
                """,
                (run_id, now, f"run {run_id} accepted; input_hash={input_hash}"),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else self._row_to_run(row)

    def mark_running(self, run_id: str) -> dict[str, Any] | None:
        now = utc_now()
        with self._write_lock, self._connect() as connection:
            row = connection.execute(
                "SELECT status, cancel_requested FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            if row["status"] != "pending" or row["cancel_requested"]:
                return self._row_to_run(
                    connection.execute(
                        "SELECT * FROM runs WHERE run_id = ?", (run_id,)
                    ).fetchone()
                )
            connection.execute(
                "UPDATE runs SET status = 'running', updated_at = ? WHERE run_id = ?",
                (now, run_id),
            )
        return self.get_run(run_id)

    def request_cancel(self, run_id: str) -> dict[str, Any] | None:
        now = utc_now()
        with self._write_lock, self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            if row["status"] in {"pending", "running"}:
                connection.execute(
                    """
                    UPDATE runs
                    SET cancel_requested = 1, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (now, run_id),
                )
                self._insert_event(
                    connection,
                    run_id,
                    "cancel-requested",
                    message="cancellation requested; no significance result will be emitted",
                )
        return self.get_run(run_id)

    def complete_run(self, run_id: str, result: dict[str, Any]) -> tuple[dict[str, Any], bool] | None:
        now = utc_now()
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'completed', result_json = ?, error_json = NULL,
                    updated_at = ?
                WHERE run_id = ?
                  AND status = 'running'
                  AND cancel_requested = 0
                """,
                (self._dumps(result), now, run_id),
            )
            completed = cursor.rowcount == 1
        return self.get_run(run_id), completed

    def fail_run(self, run_id: str, error: dict[str, Any]) -> dict[str, Any] | None:
        now = utc_now()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = 'failed', error_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (self._dumps(error), now, run_id),
            )
        return self.get_run(run_id)

    def cancel_run(self, run_id: str) -> dict[str, Any] | None:
        now = utc_now()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = 'canceled',
                    error_json = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    self._dumps(
                        {
                            "code": "CANCELED",
                            "message": "permutation computation was canceled",
                            "category": "cancellation",
                            "details": {},
                        }
                    ),
                    now,
                    run_id,
                ),
            )
        return self.get_run(run_id)

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        phase: str,
        message: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        batch_size: int | None = None,
    ) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS next_seq FROM run_events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        seq = int(row["next_seq"])
        connection.execute(
            """
            INSERT INTO run_events(
                run_id, seq, created_at, phase, completed, total,
                batch_size, message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, seq, utc_now(), phase, completed, total, batch_size, message),
        )
        return seq

    def add_event(
        self,
        run_id: str,
        phase: str,
        message: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        batch_size: int | None = None,
    ) -> int:
        with self._write_lock, self._connect() as connection:
            return self._insert_event(
                connection,
                run_id,
                phase,
                message,
                completed=completed,
                total=total,
                batch_size=batch_size,
            )

    def list_events(self, run_id: str) -> list[dict[str, Any]] | None:
        with self._connect() as connection:
            run = connection.execute(
                "SELECT run_id FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                return None
            rows = connection.execute(
                """
                SELECT run_id, seq, created_at, phase, completed, total,
                       batch_size, message
                FROM run_events
                WHERE run_id = ?
                ORDER BY seq
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def reset_stale_runs(self) -> int:
        now = utc_now()
        error = self._dumps(
            {
                "code": "SERVICE_INTERRUPTED",
                "message": "run was interrupted before the service restarted",
                "category": "unknown",
                "details": {},
            }
        )
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'failed', error_json = ?, updated_at = ?
                WHERE status IN ('pending', 'running')
                """,
                (error, now),
            )
            return cursor.rowcount
