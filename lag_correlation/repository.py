"""SQLite persistence for versioned analysis results and diagnostics."""

from __future__ import annotations

import json
import math
import os
import sqlite3
from typing import Any

from .config import ALGORITHM_VERSION, SCHEMA_VERSION
from .errors import DuplicateRequestError, ResultNotFoundError
from .kernel.analysis import CAUSALITY_NOTICE, MATCHING_RULE
from .kernel.models import AnalysisResult


class AnalysisRepository:
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        parent = os.path.dirname(os.path.abspath(database_path))
        os.makedirs(parent, exist_ok=True)
        self._initialize()

    def save(self, result: AnalysisResult) -> None:
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO analysis_run (
                        request_id, status, failure_category, reason, peak_lag,
                        peak_correlation, anchor_side, anchor_median_interval,
                        other_median_interval, minimum_pairs, detrend,
                        standardization, valid_lag_count, evaluated_lag_count,
                        schema_version, algorithm_version, matching_rule,
                        causality_notice, best_lag_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.request_id,
                        result.status,
                        result.failure_category,
                        result.reason,
                        result.peak_lag,
                        result.peak_correlation,
                        result.anchor_side,
                        result.anchor_median_interval,
                        result.other_median_interval,
                        result.minimum_pairs,
                        result.detrend,
                        result.standardization,
                        result.valid_lag_count,
                        result.evaluated_lag_count,
                        SCHEMA_VERSION,
                        ALGORITHM_VERSION,
                        MATCHING_RULE,
                        CAUSALITY_NOTICE,
                        json.dumps(
                            result.best_lag.to_dict(include_pairs=False)
                            if result.best_lag is not None
                            else None,
                            allow_nan=False,
                        ),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateRequestError(
                    f"request_id {result.request_id!r} already has a stored result"
                ) from exc

            connection.executemany(
                """
                INSERT INTO diagnostic_event (request_id, sequence, event_code, status,
                                              failure_category, lag, n_pairs, correlation,
                                              accepted, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        result.request_id,
                        sequence,
                        event.get("event_code"),
                        event.get("status"),
                        event.get("failure_category"),
                        event.get("lag"),
                        event.get("n_pairs"),
                        event.get("correlation"),
                        event.get("accepted"),
                        json.dumps(_json_safe(event), allow_nan=False),
                    )
                    for sequence, event in enumerate(result.diagnostics)
                ],
            )

    def get_response_dict(self, request_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            run = connection.execute(
                "SELECT * FROM analysis_run WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if run is None:
                raise ResultNotFoundError(f"no stored result for request_id {request_id!r}")
            events = connection.execute(
                """
                SELECT payload_json FROM diagnostic_event
                WHERE request_id = ?
                ORDER BY sequence
                """,
                (request_id,),
            ).fetchall()

        data = dict(run)
        return {
            "request_id": data["request_id"],
            "status": data["status"],
            "failure_category": data["failure_category"],
            "reason": data["reason"],
            "peak_lag": data["peak_lag"],
            "peak_correlation": data["peak_correlation"],
            "anchor_side": data["anchor_side"],
            "valid_lag_count": data["valid_lag_count"],
            "evaluated_lag_count": data["evaluated_lag_count"],
            "minimum_pairs": data["minimum_pairs"],
            "detrend": data["detrend"],
            "standardization": data["standardization"],
            "schema_version": data["schema_version"],
            "algorithm_version": data["algorithm_version"],
            "matching_rule": data["matching_rule"],
            "causality_notice": data["causality_notice"],
            "best_lag": json.loads(data["best_lag_json"]),
            "diagnostics": [json.loads(row["payload_json"]) for row in events],
        }

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_run (
                    request_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    failure_category TEXT,
                    reason TEXT NOT NULL,
                    peak_lag REAL,
                    peak_correlation REAL,
                    anchor_side TEXT NOT NULL,
                    anchor_median_interval REAL NOT NULL,
                    other_median_interval REAL NOT NULL,
                    minimum_pairs INTEGER NOT NULL,
                    detrend TEXT NOT NULL,
                    standardization TEXT NOT NULL,
                    valid_lag_count INTEGER NOT NULL,
                    evaluated_lag_count INTEGER NOT NULL,
                    schema_version TEXT NOT NULL,
                    algorithm_version TEXT NOT NULL,
                    matching_rule TEXT NOT NULL,
                    causality_notice TEXT NOT NULL,
                    best_lag_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnostic_event (
                    request_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_code TEXT NOT NULL,
                    status TEXT,
                    failure_category TEXT,
                    lag REAL,
                    n_pairs INTEGER,
                    correlation REAL,
                    accepted INTEGER,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (request_id, sequence),
                    FOREIGN KEY (request_id) REFERENCES analysis_run(request_id)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
