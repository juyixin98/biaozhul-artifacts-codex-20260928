"""SQLite-backed evidence storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import MutationReport


SCHEMA = """
CREATE TABLE IF NOT EXISTS mutation_runs (
    request_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    project_path TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS mutant_records (
    request_id TEXT NOT NULL,
    mutant_id TEXT NOT NULL,
    fate TEXT NOT NULL,
    determination TEXT NOT NULL,
    operator_family TEXT NOT NULL,
    source_path TEXT NOT NULL,
    line INTEGER NOT NULL,
    before_expr TEXT NOT NULL,
    after_expr TEXT NOT NULL,
    reason TEXT NOT NULL,
    report_json TEXT NOT NULL,
    PRIMARY KEY (request_id, mutant_id),
    FOREIGN KEY(request_id) REFERENCES mutation_runs(request_id)
);

CREATE TABLE IF NOT EXISTS run_events (
    request_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    step TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT NOT NULL,
    location TEXT,
    engine_version TEXT NOT NULL,
    PRIMARY KEY (request_id, sequence),
    FOREIGN KEY(request_id) REFERENCES mutation_runs(request_id)
);

CREATE INDEX IF NOT EXISTS idx_mutant_records_fate ON mutant_records(request_id, fate);
"""


class EvidenceStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def save_report(self, report: MutationReport) -> None:
        payload = report.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO mutation_runs
                    (request_id, status, project_path, engine_version, report_json, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.request_id,
                    report.status,
                    report.project_path,
                    report.engine_version,
                    payload,
                    report.created_at,
                    report.completed_at,
                ),
            )
            connection.execute("DELETE FROM mutant_records WHERE request_id = ?", (report.request_id,))
            connection.execute("DELETE FROM run_events WHERE request_id = ?", (report.request_id,))
            connection.executemany(
                """
                INSERT INTO mutant_records (
                    request_id, mutant_id, fate, determination, operator_family,
                    source_path, line, before_expr, after_expr, reason, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        report.request_id,
                        item.mutant_id,
                        item.fate,
                        item.determination,
                        item.operator_family,
                        item.location.path,
                        item.location.line,
                        item.before,
                        item.after,
                        item.reason,
                        item.model_dump_json(),
                    )
                    for item in report.mutants
                ],
            )
            connection.executemany(
                """
                INSERT INTO run_events (
                    request_id, sequence, timestamp, step, status, detail, location, engine_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        report.request_id,
                        event.sequence,
                        event.timestamp,
                        event.step,
                        event.status,
                        event.detail,
                        event.location,
                        event.engine_version,
                    )
                    for event in report.events
                ],
            )

    def get_report(self, request_id: str) -> MutationReport | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT report_json FROM mutation_runs WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return MutationReport.model_validate(json.loads(row["report_json"]))

    def list_reports(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT request_id, status, project_path, engine_version, created_at, completed_at
                FROM mutation_runs
                ORDER BY created_at DESC, request_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
