"""SQLite-backed request journal for local audit and reproducibility."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.config import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS request_journal (
    request_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    methods_json TEXT NOT NULL,
    n_samples INTEGER,
    n_features INTEGER,
    payload_json TEXT NOT NULL,
    response_json TEXT,
    failure_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_request_journal_created_at
ON request_journal(created_at);
"""


class RequestJournal:
    """Thin SQLite adapter with all database access isolated to one module."""

    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path = Path(database_path or settings.database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA_SQL)

    def record(
        self,
        *,
        request_id: str,
        created_at: str,
        status: str,
        methods: list[str],
        n_samples: int | None,
        n_features: int | None,
        payload: dict[str, Any],
        response: dict[str, Any] | None,
        failure: dict[str, Any] | None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO request_journal (
                    request_id, created_at, status, methods_json, n_samples,
                    n_features, payload_json, response_json, failure_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    created_at,
                    status,
                    json.dumps(methods, sort_keys=True, ensure_ascii=False),
                    n_samples,
                    n_features,
                    json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str),
                    json.dumps(response, sort_keys=True, ensure_ascii=False, default=str)
                    if response is not None
                    else None,
                    json.dumps(failure, sort_keys=True, ensure_ascii=False, default=str)
                    if failure is not None
                    else None,
                ),
            )

    def fetch(self, request_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM request_journal WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        for key in ["methods_json", "payload_json", "response_json", "failure_json"]:
            item[key] = json.loads(item[key]) if item[key] is not None else None
        return item
