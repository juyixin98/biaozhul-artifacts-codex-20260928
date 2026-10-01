from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_database_url(value: str) -> str:
    if not value.startswith("sqlite:"):
        raise ValueError("Only local SQLite database URLs are supported")
    raw_path = value.removeprefix("sqlite:///").removeprefix("sqlite://")
    if raw_path == ":memory:" or not raw_path:
        return ":memory:"
    return str(Path(raw_path))


@dataclass(frozen=True)
class Settings:
    database_path: str = "data/analysis_results.sqlite3"
    log_level: str = "INFO"
    default_confidence: float = 0.95
    default_imputations: int = 5
    default_burn_in: int = 10

    @property
    def database_url(self) -> str:
        if self.database_path == ":memory:":
            return "sqlite:///:memory:"
        return f"sqlite:///{self.database_path}"


def load_settings(env_file: str | os.PathLike[str] | None = None) -> Settings:
    values = {
        "SURVEY_DATABASE_URL": "sqlite:///./data/analysis_results.sqlite3",
        "SURVEY_LOG_LEVEL": "INFO",
        "SURVEY_DEFAULT_CONFIDENCE": "0.95",
        "SURVEY_DEFAULT_IMPUTATIONS": "5",
        "SURVEY_DEFAULT_BURN_IN": "10",
    }
    path = Path(env_file) if env_file is not None else Path("config/default.env")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    values.update({k: v for k, v in os.environ.items() if k in values})

    database_path = _parse_database_url(values["SURVEY_DATABASE_URL"])
    confidence = float(values["SURVEY_DEFAULT_CONFIDENCE"])
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence level must be between 0 and 1")
    imputations = int(values["SURVEY_DEFAULT_IMPUTATIONS"])
    if imputations < 2:
        raise ValueError("multiple imputation requires at least two imputations")
    burn_in = int(values["SURVEY_DEFAULT_BURN_IN"])
    if burn_in < 1:
        raise ValueError("burn-in must be at least one")
    return Settings(
        database_path=database_path,
        log_level=values["SURVEY_LOG_LEVEL"].upper(),
        default_confidence=confidence,
        default_imputations=imputations,
        default_burn_in=burn_in,
    )
