"""SQLite persistence for immutable succeeded calibration runs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import ValidationError

from .contracts import RunResult, RunSummary
from .errors import ConflictError, PersistenceError


class RunRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        try:
            with self.connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS calibration_runs (
                        run_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        schema_version TEXT NOT NULL,
                        input_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        result_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_calibration_runs_created "
                    "ON calibration_runs(created_at)"
                )
        except sqlite3.DatabaseError as exc:
            raise PersistenceError(f"could not initialize SQLite database: {exc}") from exc

    def save(self, result: RunResult) -> None:
        try:
            with self.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO calibration_runs
                        (run_id, status, schema_version, input_hash, created_at, result_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.run_id,
                        result.status,
                        result.versions.schema_version,
                        result.input_hash,
                        result.versions.generated_at,
                        result.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(
                f"run_id already exists: {result.run_id}",
                details={"run_id": result.run_id},
            ) from exc
        except sqlite3.DatabaseError as exc:
            raise PersistenceError(f"could not persist calibration run: {exc}") from exc

    def get(self, run_id: str) -> RunResult:
        try:
            with self.connect() as connection:
                row = connection.execute(
                    "SELECT result_json FROM calibration_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise PersistenceError(f"could not read calibration run: {exc}") from exc
        if row is None:
            raise PersistenceError(
                f"run_id not found: {run_id}",
                code="run_not_found",
                status_code=404,
                details={"run_id": run_id},
            )
        try:
            return RunResult.model_validate_json(row["result_json"])
        except ValidationError as exc:
            raise PersistenceError("stored result failed schema validation") from exc

    def list_summaries(self) -> list[RunSummary]:
        try:
            with self.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT run_id, status, schema_version, input_hash, created_at
                    FROM calibration_runs
                    ORDER BY created_at DESC, run_id ASC
                    """
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise PersistenceError(f"could not list calibration runs: {exc}") from exc
        return [RunSummary.model_validate(dict(row)) for row in rows]
