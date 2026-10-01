"""SQLite persistence for versioned analyses and explainable event logs."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .versioning import SCHEMA_MIGRATION_VERSION


class AnalysisNotFound(KeyError):
    pass


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    client_request_id TEXT,
    analysis_name TEXT,
    created_at TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    service_version TEXT NOT NULL,
    kernel_version TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analyses_request_id ON analyses(request_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);

CREATE TABLE IF NOT EXISTS analysis_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    component TEXT NOT NULL,
    step TEXT NOT NULL,
    status TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    UNIQUE(analysis_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_analysis_events_analysis ON analysis_events(analysis_id, sequence);

CREATE TABLE IF NOT EXISTS http_logs (
    request_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status_code INTEGER,
    elapsed_ms REAL,
    detail_json TEXT NOT NULL
);
"""


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class SurvivalRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        directory = os.path.dirname(os.path.abspath(db_path))
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(f"PRAGMA user_version = {SCHEMA_MIGRATION_VERSION}")
            connection.executescript(SCHEMA_SQL)

    def save_analysis(self, request: dict, result: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO analyses (
                    analysis_id, request_id, client_request_id, analysis_name,
                    created_at, schema_version, service_version, kernel_version,
                    request_json, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result["analysis_id"],
                    result["request_id"],
                    request.get("client_request_id"),
                    result.get("analysis_name"),
                    result["created_at"],
                    result["schema_version"],
                    result["service_version"],
                    result["kernel_version"],
                    _dumps(request),
                    _dumps(result),
                ),
            )
            self._replace_events(connection, result["analysis_id"], result["trace"])

    def get_analysis(self, analysis_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT response_json FROM analyses WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        if row is None:
            raise AnalysisNotFound(analysis_id)
        return json.loads(row["response_json"])

    def get_events(self, analysis_id: str) -> list[dict]:
        with self.connect() as connection:
            analysis = connection.execute(
                "SELECT request_id FROM analyses WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
            if analysis is None:
                raise AnalysisNotFound(analysis_id)
            rows = connection.execute(
                """
                SELECT sequence, timestamp, component, step, status, detail_json
                FROM analysis_events
                WHERE analysis_id = ?
                ORDER BY sequence
                """,
                (analysis_id,),
            ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "timestamp": row["timestamp"],
                "component": row["component"],
                "step": row["step"],
                "status": row["status"],
                "detail": json.loads(row["detail_json"]),
            }
            for row in rows
        ]

    def insert_http_log(
        self,
        *,
        request_id: str,
        started_at: str,
        method: str,
        path: str,
        detail: dict,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO http_logs (
                    request_id, started_at, method, path, detail_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (request_id, started_at, method, path, _dumps(detail)),
            )

    def finish_http_log(
        self,
        *,
        request_id: str,
        finished_at: str,
        status_code: int,
        elapsed_ms: float,
        detail: dict,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE http_logs
                SET finished_at = ?, status_code = ?, elapsed_ms = ?, detail_json = ?
                WHERE request_id = ?
                """,
                (finished_at, status_code, elapsed_ms, _dumps(detail), request_id),
            )

    @staticmethod
    def _replace_events(
        connection: sqlite3.Connection, analysis_id: str, events: Iterable[dict]
    ) -> None:
        connection.execute("DELETE FROM analysis_events WHERE analysis_id = ?", (analysis_id,))
        SurvivalRepository._insert_events(connection, analysis_id, events)

    @staticmethod
    def _insert_events(
        connection: sqlite3.Connection, analysis_id: str, events: Iterable[dict]
    ) -> None:
        for sequence, event in enumerate(events):
            connection.execute(
                """
                INSERT INTO analysis_events (
                    analysis_id, sequence, timestamp, component, step, status, detail_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    sequence,
                    event["timestamp"],
                    event["component"],
                    event["step"],
                    event["status"],
                    _dumps(event["detail"]),
                ),
            )
