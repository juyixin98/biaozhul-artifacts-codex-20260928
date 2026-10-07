from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from typing import Any

from .artifacts import CalibrationArtifact


class SqliteModelRepository:
    def __init__(self, path: str) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        if self.path != ":memory:":
            parent = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(parent, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    score_name TEXT NOT NULL,
                    target_name TEXT NOT NULL,
                    extrapolation TEXT NOT NULL,
                    artifact_schema_version TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    artifact_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    module TEXT NOT NULL,
                    processing_location TEXT NOT NULL,
                    version TEXT NOT NULL,
                    failure_codes TEXT NOT NULL,
                    duration_ms REAL,
                    detail_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_request_logs_request_id
                ON request_logs(request_id);
                """
            )

    def save_model(self, artifact: CalibrationArtifact) -> None:
        payload = artifact.to_json()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO models (
                    model_id, score_name, target_name, extrapolation,
                    artifact_schema_version, created_at_utc, artifact_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.model_id,
                    artifact.score_name,
                    artifact.target_name,
                    artifact.extrapolation,
                    artifact.artifact_schema_version,
                    artifact.created_at_utc,
                    payload,
                ),
            )

    def get_artifact_json(self, model_id: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT artifact_json FROM models WHERE model_id = ?",
                (model_id,),
            ).fetchone()
        return None if row is None else str(row["artifact_json"])

    def list_models(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT model_id, score_name, target_name, extrapolation,
                       artifact_schema_version, created_at_utc
                FROM models
                ORDER BY created_at_utc DESC, model_id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def log_request(self, entry: dict[str, Any]) -> None:
        failure_codes = entry.get("failure_codes") or []
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO request_logs (
                    request_id, method, path, status, module, processing_location,
                    version, failure_codes, duration_ms, detail_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["request_id"],
                    entry["method"],
                    entry["path"],
                    entry["status"],
                    entry["module"],
                    entry["processing_location"],
                    entry["version"],
                    json.dumps(list(failure_codes), sort_keys=True),
                    entry.get("duration_ms"),
                    json.dumps(entry.get("detail", {}), sort_keys=True, ensure_ascii=False),
                    entry["created_at_utc"],
                ),
            )

    def get_request_logs(self, request_id: str) -> Iterable[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM request_logs WHERE request_id = ? ORDER BY id ASC",
                    (request_id,),
                ).fetchall()
            )
