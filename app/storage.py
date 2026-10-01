"""SQLite persistence for replayable decomposition runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.errors import StateConflictError, StorageError
from app.serialization import to_jsonable


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS run_log (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    status TEXT NOT NULL CHECK (status IN ('success', 'failure')),
    error_category TEXT,
    error_message TEXT,
    request_json TEXT NOT NULL,
    result_json TEXT,
    config_fingerprint TEXT
);
CREATE TABLE IF NOT EXISTS run_trace (
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    stage TEXT NOT NULL,
    outer_iteration INTEGER NOT NULL,
    inner_iteration INTEGER NOT NULL,
    trace_json TEXT NOT NULL,
    PRIMARY KEY (run_id, step),
    FOREIGN KEY(run_id) REFERENCES run_log(run_id)
);
"""


class RunRepository:
    def __init__(self, database_path: str | Path = "runs/stl_runs.db") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._initialize()
        except sqlite3.Error as exc:
            raise StorageError(
                "unable to initialize the run database",
                {"path": str(self.database_path)},
            ) from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path, timeout=5.0, check_same_thread=False
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def exists(self, run_id: str) -> bool:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT 1 FROM run_log WHERE run_id = ?", (run_id,)
                ).fetchone()
            return row is not None
        except sqlite3.Error as exc:
            raise StorageError(
                "unable to query run state", {"run_id": run_id}
            ) from exc

    def ensure_absent(self, run_id: str) -> None:
        if self.exists(run_id):
            raise StateConflictError(
                "a run with this run_id already exists", {"run_id": run_id}
            )

    def save_failure(
        self,
        run_id: str,
        request: dict[str, Any],
        category: str,
        message: str,
        details: dict[str, Any],
    ) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO run_log(
                        run_id, status, error_category, error_message,
                        request_json, result_json, config_fingerprint
                    ) VALUES (?, 'failure', ?, ?, ?, NULL, NULL)
                    ON CONFLICT(run_id) DO NOTHING
                    """,
                    (
                        run_id,
                        category,
                        message,
                        json.dumps(to_jsonable(request), sort_keys=True),
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError(
                "unable to persist failed run", {"run_id": run_id}
            ) from exc

    def save_success(
        self,
        run_id: str,
        request: dict[str, Any],
        result: dict[str, Any],
        trace: list[dict[str, Any]],
        config_fingerprint: str,
    ) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO run_log(
                        run_id, status, error_category, error_message,
                        request_json, result_json, config_fingerprint
                    ) VALUES (?, 'success', NULL, NULL, ?, ?, ?)
                    """,
                    (
                        run_id,
                        json.dumps(to_jsonable(request), sort_keys=True),
                        json.dumps(to_jsonable(result), sort_keys=True),
                        config_fingerprint,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO run_trace(
                        run_id, step, stage, outer_iteration,
                        inner_iteration, trace_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            index,
                            item.get("stage", "unknown"),
                            int(item.get("outer_iteration", 0)),
                            int(item.get("inner_iteration", 0)),
                            json.dumps(to_jsonable(item), sort_keys=True),
                        )
                        for index, item in enumerate(trace)
                    ],
                )
        except sqlite3.IntegrityError as exc:
            raise StateConflictError(
                "a run with this run_id already exists", {"run_id": run_id}
            ) from exc
        except sqlite3.Error as exc:
            raise StorageError(
                "unable to persist successful run", {"run_id": run_id}
            ) from exc

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM run_log WHERE run_id = ?", (run_id,)
                ).fetchone()
                if row is None:
                    return None
                trace_rows = connection.execute(
                    """
                    SELECT trace_json FROM run_trace
                    WHERE run_id = ? ORDER BY step
                    """,
                    (run_id,),
                ).fetchall()
            result = dict(row)
            result["trace"] = [json.loads(row["trace_json"]) for row in trace_rows]
            return result
        except sqlite3.Error as exc:
            raise StorageError(
                "unable to read run", {"run_id": run_id}
            ) from exc

