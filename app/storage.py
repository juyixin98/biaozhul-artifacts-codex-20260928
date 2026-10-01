"""SQLite persistence for immutable run and batch versions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.errors import (
    ComputationError,
    NotFoundError,
    ResourceExhaustedError,
    StateConflictError,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    alpha REAL NOT NULL,
    beta REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_versions (
    batch_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    batch_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    successes INTEGER NOT NULL CHECK (successes >= 0),
    failures INTEGER NOT NULL CHECK (failures >= 0),
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded')),
    reason TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, batch_id, version)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_batch_version
ON batch_versions(run_id, batch_id)
WHERE status = 'active';

CREATE TABLE IF NOT EXISTS run_versions (
    run_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL
        CHECK (event_type IN ('created', 'batch_submitted', 'batch_revised')),
    batch_version_id INTEGER REFERENCES batch_versions(batch_version_id),
    posterior_alpha REAL NOT NULL,
    posterior_beta REAL NOT NULL,
    total_successes INTEGER NOT NULL,
    total_failures INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    UNIQUE (run_id, sequence)
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    if "metadata_json" in data:
        try:
            data["metadata"] = json.loads(data.pop("metadata_json"))
        except json.JSONDecodeError as exc:
            raise ComputationError(
                "Stored run version metadata is not valid JSON.",
                {"run_version_id": data.get("run_version_id")},
            ) from exc
    return data


class SQLiteStore:
    def __init__(self, database: str | Path = ":memory:") -> None:
        self.database = str(database)
        self._owns_file = self.database != ":memory:"
        if self.database == ":memory:":
            self._memory_connection = sqlite3.connect(
                ":memory:", check_same_thread=False
            )
            self._memory_connection.row_factory = sqlite3.Row
            self._memory_connection.execute("PRAGMA foreign_keys = ON")
        else:
            self._memory_connection = None
            Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return self._memory_connection
        try:
            conn = sqlite3.connect(self.database, timeout=5.0)
        except sqlite3.Error as exc:
            raise ResourceExhaustedError(
                "SQLite could not open the local database.",
                {"database": self.database},
            ) from exc
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(SCHEMA)
        except sqlite3.OperationalError as exc:
            raise ResourceExhaustedError(
                "SQLite schema initialization failed.", {"database": self.database}
            ) from exc

    @staticmethod
    def _translate_sqlite_error(exc: sqlite3.DatabaseError) -> Exception:
        message = str(exc).lower()
        if "disk" in message or "i/o" in message or "readonly" in message:
            return ResourceExhaustedError(
                "SQLite storage resource is unavailable.", {"database_error": str(exc)}
            )
        if "database is locked" in message or "unable to open" in message:
            return ResourceExhaustedError(
                "SQLite database is temporarily unavailable.",
                {"database_error": str(exc)},
            )
        return ResourceExhaustedError(
            "SQLite storage operation failed.", {"database_error": str(exc)}
        )

    def create_run(self, run_id: str, alpha: float, beta: float) -> dict[str, Any]:
        created_at = utc_now()
        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    conn.execute(
                        "INSERT INTO runs(run_id, alpha, beta, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (run_id, alpha, beta, created_at),
                    )
                    cursor = conn.execute(
                        """
                        INSERT INTO run_versions(
                            run_id, sequence, event_type, batch_version_id,
                            posterior_alpha, posterior_beta, total_successes,
                            total_failures, created_at, metadata_json
                        ) VALUES (?, 1, 'created', NULL, ?, ?, 0, 0, ?, ?)
                        """,
                        (
                            run_id,
                            alpha,
                            beta,
                            created_at,
                            json.dumps({"prior_alpha": alpha, "prior_beta": beta}),
                        ),
                    )
                    version_id = cursor.lastrowid
                except Exception:
                    conn.rollback()
                    raise
                conn.commit()
        except sqlite3.IntegrityError as exc:
            raise StateConflictError(
                "A run with this ID already exists.", {"run_id": run_id}
            ) from exc
        except sqlite3.OperationalError as exc:
            raise self._translate_sqlite_error(exc) from exc

        return {
            "run_id": run_id,
            "alpha": alpha,
            "beta": beta,
            "created_at": created_at,
            "run_version_id": version_id,
            "sequence": 1,
        }

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        run = _row_to_dict(row)
        if run is None:
            raise NotFoundError("Run does not exist.", {"run_id": run_id})
        return run

    def active_batches(self, run_id: str) -> list[dict[str, Any]]:
        self.get_run(run_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM batch_versions
                WHERE run_id = ? AND status = 'active'
                ORDER BY batch_version_id
                """,
                (run_id,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]  # type: ignore[list-item]

    def list_batch_versions(self, run_id: str) -> list[dict[str, Any]]:
        self.get_run(run_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM batch_versions
                WHERE run_id = ?
                ORDER BY batch_id, version
                """,
                (run_id,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]  # type: ignore[list-item]

    def list_run_versions(self, run_id: str) -> list[dict[str, Any]]:
        self.get_run(run_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM run_versions
                WHERE run_id = ?
                ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]  # type: ignore[list-item]

    def get_active_batch(self, run_id: str, batch_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM batch_versions
                WHERE run_id = ? AND batch_id = ? AND status = 'active'
                """,
                (run_id, batch_id),
            ).fetchone()
        return _row_to_dict(row)

    def apply_batch(
        self,
        run_id: str,
        batch_id: str,
        successes: int,
        failures: int,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Submit a new batch or atomically replace its active version."""
        self.get_run(run_id)
        now = utc_now()
        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    old = conn.execute(
                        """
                        SELECT * FROM batch_versions
                        WHERE run_id = ? AND batch_id = ? AND status = 'active'
                        """,
                        (run_id, batch_id),
                    ).fetchone()
                    if old is None:
                        version = 1
                        event_type = "batch_submitted"
                    else:
                        version = old["version"] + 1
                        event_type = "batch_revised"
                        conn.execute(
                            """
                            UPDATE batch_versions
                            SET status = 'superseded'
                            WHERE batch_version_id = ?
                            """,
                            (old["batch_version_id"],),
                        )

                    cursor = conn.execute(
                        """
                        INSERT INTO batch_versions(
                            run_id, batch_id, version, successes, failures,
                            status, reason, created_at
                        ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
                        """,
                        (
                            run_id,
                            batch_id,
                            version,
                            successes,
                            failures,
                            reason,
                            now,
                        ),
                    )
                    batch_version_id = cursor.lastrowid

                    active = conn.execute(
                        """
                        SELECT
                            COALESCE(SUM(successes), 0) AS total_successes,
                            COALESCE(SUM(failures), 0) AS total_failures
                        FROM batch_versions
                        WHERE run_id = ? AND status = 'active'
                        """,
                        (run_id,),
                    ).fetchone()
                    sequence_row = conn.execute(
                        "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence "
                        "FROM run_versions WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()
                    run = conn.execute(
                        "SELECT alpha, beta FROM runs WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()
                    post_alpha = run["alpha"] + active["total_successes"]
                    post_beta = run["beta"] + active["total_failures"]
                    run_cursor = conn.execute(
                        """
                        INSERT INTO run_versions(
                            run_id, sequence, event_type, batch_version_id,
                            posterior_alpha, posterior_beta, total_successes,
                            total_failures, created_at, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            sequence_row["next_sequence"],
                            event_type,
                            batch_version_id,
                            post_alpha,
                            post_beta,
                            active["total_successes"],
                            active["total_failures"],
                            now,
                            json.dumps(
                                {
                                    "batch_id": batch_id,
                                    "version": version,
                                    "successes": successes,
                                    "failures": failures,
                                    "reason": reason,
                                }
                            ),
                        ),
                    )
                except Exception:
                    conn.rollback()
                    raise
                conn.commit()
        except sqlite3.OperationalError as exc:
            raise self._translate_sqlite_error(exc) from exc
        except sqlite3.IntegrityError as exc:
            raise StateConflictError(
                "Batch version conflicts with existing state.",
                {"run_id": run_id, "batch_id": batch_id},
            ) from exc

        return {
            "run_id": run_id,
            "batch_id": batch_id,
            "version": version,
            "batch_version_id": batch_version_id,
            "run_version_id": run_cursor.lastrowid,
            "sequence": sequence_row["next_sequence"],
            "event_type": event_type,
            "successes": successes,
            "failures": failures,
            "reason": reason,
            "posterior_alpha": post_alpha,
            "posterior_beta": post_beta,
            "total_successes": active["total_successes"],
            "total_failures": active["total_failures"],
            "created_at": now,
        }
