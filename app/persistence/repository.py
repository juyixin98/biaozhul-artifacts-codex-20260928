from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.errors import PersistenceError, ResourceExhaustedError


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    prior_alpha REAL NOT NULL CHECK (
        prior_alpha IS NOT NULL AND typeof(prior_alpha) IN ('real', 'integer') AND prior_alpha > 0
    ),
    prior_beta REAL NOT NULL CHECK (
        prior_beta IS NOT NULL AND typeof(prior_beta) IN ('real', 'integer') AND prior_beta > 0
    ),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE RESTRICT,
    revision_of_batch_id TEXT REFERENCES batches(id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'reverted')),
    version INTEGER NOT NULL CHECK (version >= 0),
    success_count INTEGER NOT NULL CHECK (
        success_count IS NOT NULL AND typeof(success_count) = 'integer' AND success_count >= 0
    ),
    failure_count INTEGER NOT NULL CHECK (
        failure_count IS NOT NULL AND typeof(failure_count) = 'integer' AND failure_count >= 0
    ),
    reason TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_batches_model ON batches(model_id, version);
CREATE INDEX IF NOT EXISTS idx_batches_revision ON batches(revision_of_batch_id);

CREATE TABLE IF NOT EXISTS events (
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE RESTRICT,
    version INTEGER NOT NULL CHECK (version >= 0),
    event_type TEXT NOT NULL,
    batch_id TEXT REFERENCES batches(id) ON DELETE RESTRICT,
    delta_success INTEGER NOT NULL CHECK (
        delta_success IS NOT NULL AND typeof(delta_success) = 'integer'
    ),
    delta_failure INTEGER NOT NULL CHECK (
        delta_failure IS NOT NULL AND typeof(delta_failure) = 'integer'
    ),
    total_success INTEGER NOT NULL CHECK (
        total_success IS NOT NULL AND typeof(total_success) = 'integer' AND total_success >= 0
    ),
    total_failure INTEGER NOT NULL CHECK (
        total_failure IS NOT NULL AND typeof(total_failure) = 'integer' AND total_failure >= 0
    ),
    details TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (model_id, version)
);

CREATE INDEX IF NOT EXISTS idx_events_time ON events(model_id, created_at);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def dict_factory(cursor: sqlite3.Cursor, row: tuple[Any, ...]) -> dict[str, Any]:
    return {column[0]: row[index] for index, column in enumerate(cursor.description)}


class SQLiteRepository:
    def __init__(
        self,
        path: Path | str,
        *,
        connection_factory=None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection_factory = connection_factory or sqlite3.connect
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = self.connection_factory(str(self.path), timeout=10)
            connection.row_factory = dict_factory
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 10000")
            return connection
        except sqlite3.Error as exc:
            message = str(exc).lower()
            if isinstance(exc, sqlite3.OperationalError) and (
                "disk" in message or "full" in message or "space" in message
            ):
                raise ResourceExhaustedError(
                    "Local SQLite storage capacity was exhausted.",
                    {"sqlite_error": exc.__class__.__name__, "message": str(exc)},
                ) from exc
            raise PersistenceError(
                "Unable to open or initialize the SQLite database.",
                {
                    "path": str(self.path),
                    "sqlite_error": exc.__class__.__name__,
                    "message": str(exc),
                },
            ) from exc

    def _initialize(self) -> None:
        with self.transaction() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        except sqlite3.Error as exc:
            message = str(exc).lower()
            if isinstance(exc, sqlite3.OperationalError) and (
                "disk" in message or "full" in message or "space" in message
            ):
                raise ResourceExhaustedError(
                    "Local SQLite storage capacity was exhausted.",
                    {"sqlite_error": exc.__class__.__name__, "message": str(exc)},
                ) from exc
            raise PersistenceError(
                "SQLite transaction failed.",
                {"sqlite_error": exc.__class__.__name__, "message": str(exc)},
            ) from exc
        finally:
            connection.close()

    def insert_model(
        self,
        connection: sqlite3.Connection,
        *,
        model_id: str,
        prior_alpha: float,
        prior_beta: float,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO models (id, prior_alpha, prior_beta, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (model_id, float(prior_alpha), float(prior_beta), created_at),
        )

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        with self.transaction() as connection:
            return connection.execute(
                "SELECT * FROM models WHERE id = ?", (model_id,)
            ).fetchone()

    def get_batch(self, model_id: str, batch_id: str) -> dict[str, Any] | None:
        with self.transaction() as connection:
            return connection.execute(
                "SELECT * FROM batches WHERE model_id = ? AND id = ?",
                (model_id, batch_id),
            ).fetchone()

    def list_batches(self, model_id: str) -> list[dict[str, Any]]:
        with self.transaction() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM batches WHERE model_id = ? ORDER BY version, rowid",
                    (model_id,),
                ).fetchall()
            )

    def list_active_batches(
        self, connection: sqlite3.Connection, model_id: str
    ) -> list[dict[str, Any]]:
        return list(
            connection.execute(
                "SELECT * FROM batches WHERE model_id = ? AND status = 'active' ORDER BY rowid",
                (model_id,),
            ).fetchall()
        )

    def insert_batch(
        self,
        connection: sqlite3.Connection,
        *,
        batch_id: str,
        model_id: str,
        revision_of_batch_id: str | None,
        status: str,
        version: int,
        success_count: int,
        failure_count: int,
        reason: str | None,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO batches (
                id, model_id, revision_of_batch_id, status, version,
                success_count, failure_count, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                model_id,
                revision_of_batch_id,
                status,
                version,
                success_count,
                failure_count,
                reason,
                created_at,
            ),
        )

    def update_batch_status(
        self, connection: sqlite3.Connection, batch_id: str, status: str
    ) -> None:
        connection.execute(
            "UPDATE batches SET status = ? WHERE id = ?", (status, batch_id)
        )

    def insert_event(
        self,
        connection: sqlite3.Connection,
        *,
        model_id: str,
        version: int,
        event_type: str,
        batch_id: str | None,
        delta_success: int,
        delta_failure: int,
        total_success: int,
        total_failure: int,
        details: dict[str, Any],
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO events (
                model_id, version, event_type, batch_id, delta_success, delta_failure,
                total_success, total_failure, details, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                version,
                event_type,
                batch_id,
                delta_success,
                delta_failure,
                total_success,
                total_failure,
                json.dumps(details, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
                created_at,
            ),
        )

    def list_events(self, model_id: str) -> list[dict[str, Any]]:
        with self.transaction() as connection:
            rows = list(
                connection.execute(
                    "SELECT * FROM events WHERE model_id = ? ORDER BY version", (model_id,)
                ).fetchall()
            )
        for row in rows:
            row["details"] = json.loads(row["details"])
        return rows
