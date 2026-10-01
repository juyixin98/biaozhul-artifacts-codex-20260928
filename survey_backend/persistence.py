from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from survey_backend.models import AnalysisGroupResponse, AnalysisResponse, AnalysisSummary


class ResultStore:
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            database_path,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_groups (
                    analysis_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_versions (
                    analysis_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    dataset_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (analysis_id, version),
                    FOREIGN KEY (analysis_id) REFERENCES analysis_groups(analysis_id)
                )
                """
            )
            self._connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_version_hash
                ON analysis_versions(analysis_id, request_hash)
                """
            )

    def save(self, response: AnalysisResponse) -> AnalysisResponse:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO analysis_groups(analysis_id, created_at) VALUES (?, ?)",
                (response.analysis_id, created_at),
            )
            existing = self._connection.execute(
                """
                SELECT version, payload FROM analysis_versions
                WHERE analysis_id = ? AND request_hash = ?
                """,
                (response.analysis_id, response.request_hash),
            ).fetchone()
            if existing is not None:
                return AnalysisResponse.model_validate_json(existing["payload"])
            version_row = self._connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) AS latest
                FROM analysis_versions WHERE analysis_id = ?
                """,
                (response.analysis_id,),
            ).fetchone()
            version = int(version_row["latest"]) + 1
            response = response.model_copy(update={"version": version})
            self._connection.execute(
                """
                INSERT INTO analysis_versions(
                    analysis_id, version, dataset_id, request_id,
                    request_hash, created_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response.analysis_id,
                    response.version,
                    response.dataset_id,
                    response.request_id,
                    response.request_hash,
                    created_at,
                    response.model_dump_json(),
                ),
            )
        return response

    def get(self, analysis_id: str, version: int | None = None) -> AnalysisResponse | None:
        if version is None:
            row = self._connection.execute(
                """
                SELECT payload FROM analysis_versions
                WHERE analysis_id = ?
                ORDER BY version DESC LIMIT 1
                """,
                (analysis_id,),
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT payload FROM analysis_versions WHERE analysis_id = ? AND version = ?",
                (analysis_id, version),
            ).fetchone()
        return None if row is None else AnalysisResponse.model_validate_json(row["payload"])

    def list_versions(self, analysis_id: str) -> AnalysisGroupResponse | None:
        group = self._connection.execute(
            "SELECT analysis_id FROM analysis_groups WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()
        if group is None:
            return None
        rows = self._connection.execute(
            """
            SELECT version, dataset_id, request_id, created_at
            FROM analysis_versions
            WHERE analysis_id = ?
            ORDER BY version
            """,
            (analysis_id,),
        ).fetchall()
        return AnalysisGroupResponse(
            analysis_id=analysis_id,
            latest_version=int(rows[-1]["version"]) if rows else 0,
            versions=[
                AnalysisSummary(
                    analysis_id=analysis_id,
                    version=int(row["version"]),
                    dataset_id=row["dataset_id"],
                    request_id=row["request_id"],
                    created_at=row["created_at"],
                )
                for row in rows
            ],
        )

    def close(self) -> None:
        self._connection.close()
