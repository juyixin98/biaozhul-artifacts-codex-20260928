from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str = "bounded-trust-region-nls"
    environment: str = "local"
    log_level: str = "INFO"
    log_json: bool = True
    default_derivative_tolerance: float = 5e-5
    default_relative_step: float = 1e-7


def get_settings() -> Settings:
    return Settings(
        service_name=os.getenv("NLS_SERVICE_NAME", "bounded-trust-region-nls"),
        environment=os.getenv("NLS_ENVIRONMENT", "local"),
        log_level=os.getenv("NLS_LOG_LEVEL", "INFO").upper(),
        log_json=os.getenv("NLS_LOG_JSON", "1") not in {"0", "false", "False"},
        default_derivative_tolerance=float(
            os.getenv("NLS_DERIVATIVE_TOLERANCE", "5e-5")
        ),
        default_relative_step=float(os.getenv("NLS_RELATIVE_STEP", "1e-7")),
    )
