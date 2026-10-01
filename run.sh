#!/usr/bin/env bash
set -euo pipefail

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

exec python3 -m uvicorn app.main:app --host "$HOST" --port "$PORT" --log-level "${LOG_LEVEL,,}"
