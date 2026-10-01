"""SQLite persistence for versioned estimates and reproducibility summaries."""

from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
from pathlib import Path
from typing import Any

from .contracts import AnalysisResponse, Diagnostic, ImputedValue, PointResult
from .errors import ResultAlreadyExistsError, ResultNotFoundError
from .versioning import CONTRACT_VERSION, CORE_VERSION, STORAGE_VERSION


SCHEMA = """
CREATE TABLE IF NOT EXISTS result_versions (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE,
    contract_version TEXT NOT NULL,
    core_version TEXT NOT NULL,
    storage_version TEXT NOT NULL,
    method TEXT NOT NULL,
    model_version TEXT NOT NULL,
    estimate_json TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL,
    imputations_preview_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    n_records INTEGER NOT NULL,
    mechanism TEXT NOT NULL,
    method TEXT NOT NULL,
    model_version TEXT NOT NULL,
    estimate REAL NOT NULL,
    variance REAL NOT NULL,
    standard_error REAL NOT NULL,
    ci_low REAL NOT NULL,
    ci_high REAL NOT NULL,
    n_missing INTEGER NOT NULL,
    imputation_count INTEGER,
    target_value REAL,
    bias REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_experiment_request ON experiment_runs(request_id);
CREATE INDEX IF NOT EXISTS idx_experiment_seed_mechanism ON experiment_runs(seed, mechanism);
"""


@dataclass(frozen=True)
class StoredResult:
    request_id: str
    response: AnalysisResponse
    created_at: str


class SQLiteResultRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(experiment_runs)").fetchall()
            }
            if "imputation_count" not in columns:
                connection.execute(
                    "ALTER TABLE experiment_runs ADD COLUMN imputation_count INTEGER"
                )

    def save_response(self, response: AnalysisResponse) -> int:
        try:
            with self.connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO result_versions (
                        request_id, contract_version, core_version, storage_version,
                        method, model_version, estimate_json, diagnostics_json,
                        imputations_preview_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        response.request_id,
                        response.contract_version,
                        response.core_version,
                        STORAGE_VERSION,
                        response.estimate.method,
                        response.estimate.model_version,
                        response.estimate.model_dump_json(),
                        json.dumps([item.model_dump() for item in response.diagnostics], ensure_ascii=True),
                        json.dumps(
                            [item.model_dump() for item in response.imputations_preview],
                            ensure_ascii=True,
                        ),
                    ),
                )
                return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ResultAlreadyExistsError(
                f"a result already exists for request_id={response.request_id}"
            ) from exc

    def get_response(self, request_id: str) -> StoredResult:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM result_versions WHERE request_id = ?", (request_id,)
            ).fetchone()
        if row is None:
            raise ResultNotFoundError(f"no stored result for request_id={request_id}")
        payload = json.loads(row["estimate_json"])
        diagnostics = [Diagnostic(**item) for item in json.loads(row["diagnostics_json"])]
        imputations_preview = [
            ImputedValue(**item) for item in json.loads(row["imputations_preview_json"])
        ]
        response = AnalysisResponse(
            contract_version=row["contract_version"],
            core_version=row["core_version"],
            request_id=row["request_id"],
            estimate=PointResult(**payload),
            diagnostics=diagnostics,
            imputations_preview=imputations_preview,
        )
        return StoredResult(response.request_id, response, row["created_at"])

    def save_experiment_row(self, row: dict[str, Any]) -> None:
        allowed = {
            "request_id",
            "seed",
            "n_records",
            "mechanism",
            "method",
            "model_version",
            "estimate",
            "variance",
            "standard_error",
            "ci_low",
            "ci_high",
            "n_missing",
            "imputation_count",
            "target_value",
            "bias",
        }
        missing = allowed - row.keys()
        if missing:
            raise ValueError(f"experiment row missing keys: {sorted(missing)}")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO experiment_runs (
                    request_id, seed, n_records, mechanism, method, model_version,
                    estimate, variance, standard_error, ci_low, ci_high, n_missing,
                    imputation_count, target_value, bias
                ) VALUES (
                    :request_id, :seed, :n_records, :mechanism, :method,
                    :model_version, :estimate, :variance, :standard_error,
                    :ci_low, :ci_high, :n_missing, :imputation_count, :target_value, :bias
                )
                """,
                {key: row[key] for key in sorted(allowed)},
            )

    def list_experiment_rows(self, request_id: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if request_id is None:
                rows = connection.execute(
                    "SELECT * FROM experiment_runs ORDER BY run_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM experiment_runs WHERE request_id = ? ORDER BY run_id",
                    (request_id,),
                ).fetchall()
        return [dict(row) for row in rows]


__all__ = ["CONTRACT_VERSION", "CORE_VERSION", "SQLiteResultRepository", "StoredResult"]
