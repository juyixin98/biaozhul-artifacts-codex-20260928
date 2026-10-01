"""Version envelope and SQLite persistence for diagnostic results."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy
import scipy

from . import __version__

SCHEMA_VERSION = "mcmc-diagnostics/v1"
PROCESSING_LOCATION = "local-fastapi-sqlite"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:24]}"


def build_versioned_result(
    result: dict[str, Any],
    *,
    request_id: str,
    processing_location: str = PROCESSING_LOCATION,
) -> dict[str, Any]:
    """Attach correlation, implementation version, time, and location."""
    return {
        "request_id": request_id,
        "schema_version": SCHEMA_VERSION,
        "service_version": __version__,
        "processed_at": utc_now_iso(),
        "processing_location": processing_location,
        "dependency_versions": {
            "python_runtime": "3",
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "result": result,
    }


class ResultRepository:
    """Small SQLite repository for versioned diagnostics and request logs."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS analysis_runs (
                request_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                service_version TEXT NOT NULL,
                status TEXT NOT NULL,
                chain_count INTEGER NOT NULL,
                draws_per_chain INTEGER NOT NULL,
                processed_at TEXT NOT NULL,
                processing_location TEXT NOT NULL,
                result_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS request_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                stage TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_request_events_request_id
                ON request_events(request_id);
            """
        )
        self.connection.commit()

    def save(self, envelope: dict[str, Any]) -> None:
        result = envelope["result"]
        summary = result["input_summary"]
        self.connection.execute(
            """
            INSERT OR REPLACE INTO analysis_runs (
                request_id, schema_version, service_version, status,
                chain_count, draws_per_chain, processed_at,
                processing_location, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                envelope["request_id"],
                envelope["schema_version"],
                envelope["service_version"],
                result["status"],
                summary["chain_count"],
                summary["draws_per_chain"],
                envelope["processed_at"],
                envelope["processing_location"],
                json.dumps(envelope, ensure_ascii=True, allow_nan=False),
            ),
        )
        self.connection.commit()

    def log_event(
        self,
        request_id: str,
        stage: str,
        level: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO request_events (
                request_id, occurred_at, stage, level, message, details_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                utc_now_iso(),
                stage,
                level,
                message,
                json.dumps(details or {}, ensure_ascii=True, allow_nan=False),
            ),
        )
        self.connection.commit()

    def get(self, request_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT result_json FROM analysis_runs WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return json.loads(row["result_json"]) if row else None

    def events(self, request_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT request_id, occurred_at, stage, level, message, details_json
            FROM request_events WHERE request_id = ? ORDER BY id
            """,
            (request_id,),
        ).fetchall()
        return [
            {
                "request_id": row["request_id"],
                "occurred_at": row["occurred_at"],
                "stage": row["stage"],
                "level": row["level"],
                "message": row["message"],
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        ]

    def close(self) -> None:
        self.connection.close()
