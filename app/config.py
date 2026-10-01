"""Application constants and environment configuration."""

from __future__ import annotations

import os

SERVICE_NAME = "exact-nsga-backend"
SERVICE_VERSION = "1.0.0"
ALGORITHM_VERSION = "exact-nsga-nds/1.0.0"
DEFAULT_HOST = os.getenv("HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("PORT", "8000"))
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_OBJECTIVES = 64
