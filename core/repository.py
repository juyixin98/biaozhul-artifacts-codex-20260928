"""SQLite persistence for versioned runs, bins, and fitted calibrators."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from .errors import RunNotFoundError

ALGORITHM_VERSION = "calibration-core-0.1.0"
SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


class RunRepository:
    def __init__(self, path: str) -> None:
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.init_schema()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def init_schema(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO schema_meta(key, value)
                VALUES ('schema_version', ?)
                """,
                (str(SCHEMA_VERSION),),
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('success', 'failed')),
                    input_hash TEXT NOT NULL,
                    algorithm_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error_category TEXT,
                    error_message TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_bins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    bin_index INTEGER NOT NULL,
                    lower REAL NOT NULL,
                    upper REAL NOT NULL,
                    count INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    weight_fraction REAL NOT NULL,
                    effective_sample_size REAL NOT NULL,
                    observed_rate REAL NOT NULL,
                    mean_prediction REAL NOT NULL,
                    brier REAL NOT NULL,
                    binned_brier REAL NOT NULL,
                    empty INTEGER NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id),
                    UNIQUE(run_id, stage, bin_index)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS calibrators (
                    run_id TEXT PRIMARY KEY,
                    edges_json TEXT NOT NULL,
                    rates_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )

    def save_success(
        self,
        *,
        run_id: str,
        run_type: str,
        input_hash: str,
        created_at: str,
        request: Any,
        result: Any,
        analyses: dict[str, Any] | None = None,
        calibrator: Any | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, run_type, status, input_hash, algorithm_version,
                    created_at, request_json, result_json, error_category, error_message
                ) VALUES (?, ?, 'success', ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    run_id,
                    run_type,
                    input_hash,
                    ALGORITHM_VERSION,
                    created_at,
                    dumps(request),
                    dumps(result),
                ),
            )
            for stage, analysis in (analyses or {}).items():
                self._insert_bins(connection, run_id, stage, analysis)
            if calibrator is not None:
                connection.execute(
                    """
                    INSERT INTO calibrators(run_id, edges_json, rates_json)
                    VALUES (?, ?, ?)
                    """,
                    (run_id, dumps(calibrator.edges.tolist()), dumps(list(calibrator.bin_predicted_rates))),
                )

    def save_failure(
        self,
        *,
        run_id: str,
        run_type: str,
        input_hash: str,
        created_at: str,
        request: Any,
        category: str,
        message: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, run_type, status, input_hash, algorithm_version,
                    created_at, request_json, result_json, error_category, error_message
                ) VALUES (?, ?, 'failed', ?, ?, ?, ?, NULL, ?, ?)
                """,
                (run_id, run_type, input_hash, ALGORITHM_VERSION, created_at, dumps(request), category, message),
            )

    def _insert_bins(self, connection: sqlite3.Connection, run_id: str, stage: str, analysis: Any) -> None:
        for bin_stats in analysis.bins:
            connection.execute(
                """
                INSERT INTO run_bins(
                    run_id, stage, bin_index, lower, upper, count, weight,
                    weight_fraction, effective_sample_size, observed_rate,
                    mean_prediction, brier, binned_brier, empty
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    stage,
                    bin_stats.bin_index,
                    bin_stats.lower,
                    bin_stats.upper,
                    bin_stats.count,
                    bin_stats.weight,
                    bin_stats.weight_fraction,
                    bin_stats.effective_sample_size,
                    bin_stats.observed_rate,
                    bin_stats.mean_prediction,
                    bin_stats.brier,
                    bin_stats.binned_brier,
                    1 if bin_stats.empty else 0,
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise RunNotFoundError(f"run_id not found: {run_id}")
            bins = connection.execute(
                "SELECT * FROM run_bins WHERE run_id = ? ORDER BY stage, bin_index",
                (run_id,),
            ).fetchall()
            calibrator = connection.execute(
                "SELECT * FROM calibrators WHERE run_id = ?", (run_id,)
            ).fetchone()

        result = dict(row)
        result["request"] = json.loads(result.pop("request_json"))
        result["result"] = json.loads(result.pop("result_json")) if result["result_json"] else None
        result["bins"] = [dict(item) for item in bins]
        result["calibrator"] = dict(calibrator) if calibrator else None
        return result
