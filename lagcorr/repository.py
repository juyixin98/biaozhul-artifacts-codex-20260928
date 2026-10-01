"""SQLite persistence for analysis results and pair evidence."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
import os
import sqlite3
from typing import Iterator

from .kernel import AnalysisOutcome


SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    request_id TEXT PRIMARY KEY,
    result_version TEXT NOT NULL,
    tolerance REAL NOT NULL,
    lags_json TEXT NOT NULL,
    peak_lag REAL,
    peak_correlation REAL,
    peak_kind TEXT,
    warning TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status_summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lag_results (
    request_id TEXT NOT NULL,
    lag REAL NOT NULL,
    status TEXT NOT NULL,
    reason_code TEXT,
    reason TEXT,
    correlation REAL,
    pair_count INTEGER NOT NULL,
    diagnosis_json TEXT NOT NULL,
    PRIMARY KEY (request_id, lag),
    FOREIGN KEY (request_id) REFERENCES analyses(request_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pair_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    lag REAL NOT NULL,
    pair_order INTEGER NOT NULL,
    a_index INTEGER NOT NULL,
    b_index INTEGER NOT NULL,
    a_time REAL NOT NULL,
    b_time REAL NOT NULL,
    a_value REAL NOT NULL,
    b_value REAL NOT NULL,
    aligned_b_time REAL NOT NULL,
    residual REAL NOT NULL,
    FOREIGN KEY (request_id) REFERENCES analyses(request_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pair_records_request_lag
ON pair_records(request_id, lag);
"""


class AnalysisRepository:
    def __init__(self, database_path: str):
        self.database_path = database_path
        parent = os.path.dirname(os.path.abspath(database_path))
        os.makedirs(parent, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def save_analysis(self, request_id: str, outcome: AnalysisOutcome) -> None:
        status_summary: dict[str, int] = {}
        for result in outcome.results:
            status_summary[result.status] = status_summary.get(result.status, 0) + 1

        with self.connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO analyses (
                        request_id, result_version, tolerance, lags_json,
                        peak_lag, peak_correlation, peak_kind, warning,
                        config_json, status_summary_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        outcome.result_version,
                        outcome.tolerance,
                        json.dumps(outcome.lags),
                        outcome.peak_lag,
                        outcome.peak_correlation,
                        outcome.peak_kind,
                        outcome.warning,
                        json.dumps(asdict(outcome.config)),
                        json.dumps(status_summary),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateRequestError(request_id) from exc

            for result in outcome.results:
                connection.execute(
                    """
                    INSERT INTO lag_results (
                        request_id, lag, status, reason_code, reason,
                        correlation, pair_count, diagnosis_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        result.lag,
                        result.status,
                        result.reason_code,
                        result.reason,
                        result.correlation,
                        result.pair_count,
                        json.dumps(asdict(result.diagnosis)),
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO pair_records (
                        request_id, lag, pair_order, a_index, b_index,
                        a_time, b_time, a_value, b_value,
                        aligned_b_time, residual
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            request_id,
                            result.lag,
                            order,
                            pair.a_index,
                            pair.b_index,
                            pair.a_time,
                            pair.b_time,
                            pair.a_value,
                            pair.b_value,
                            pair.aligned_b_time,
                            pair.residual,
                        )
                        for order, pair in enumerate(result.pairs)
                    ],
                )

    def get_analysis(self, request_id: str) -> dict:
        with self.connect() as connection:
            analysis = connection.execute(
                "SELECT * FROM analyses WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if analysis is None:
                raise AnalysisNotFoundError(request_id)

            lag_rows = connection.execute(
                "SELECT * FROM lag_results WHERE request_id = ? ORDER BY lag",
                (request_id,),
            ).fetchall()
            pair_rows = connection.execute(
                "SELECT * FROM pair_records WHERE request_id = ? ORDER BY lag, pair_order",
                (request_id,),
            ).fetchall()

        status_summary: dict[str, int] = {}
        for row in lag_rows:
            status_summary[row["status"]] = status_summary.get(row["status"], 0) + 1

        pairs_by_lag: dict[float, list[dict]] = {}
        for row in pair_rows:
            pairs_by_lag.setdefault(row["lag"], []).append(
                {
                    "a_index": row["a_index"],
                    "b_index": row["b_index"],
                    "a_time": row["a_time"],
                    "b_time": row["b_time"],
                    "a_value": row["a_value"],
                    "b_value": row["b_value"],
                    "aligned_b_time": row["aligned_b_time"],
                    "residual": row["residual"],
                }
            )

        results = []
        for row in lag_rows:
            results.append(
                {
                    "lag": row["lag"],
                    "status": row["status"],
                    "reason_code": row["reason_code"],
                    "reason": row["reason"],
                    "correlation": row["correlation"],
                    "pair_count": row["pair_count"],
                    "pairs": pairs_by_lag.get(row["lag"], []),
                    "diagnosis": json.loads(row["diagnosis_json"]),
                }
            )

        return {
            "request_id": analysis["request_id"],
            "result_version": analysis["result_version"],
            "tolerance": analysis["tolerance"],
            "lags": json.loads(analysis["lags_json"]),
            "peak_lag": analysis["peak_lag"],
            "peak_correlation": analysis["peak_correlation"],
            "peak_kind": analysis["peak_kind"],
            "warning": analysis["warning"],
            "config": json.loads(analysis["config_json"]),
            "status_summary": status_summary,
            "created_at": analysis["created_at"],
            "results": results,
        }


class DuplicateRequestError(Exception):
    def __init__(self, request_id: str):
        self.request_id = request_id
        super().__init__(f"duplicate request_id: {request_id}")


class AnalysisNotFoundError(Exception):
    def __init__(self, request_id: str):
        self.request_id = request_id
        super().__init__(f"unknown request_id: {request_id}")
