#!/usr/bin/env bash
# Local service entry point. Usage: ./run_server.sh [port]
set -euo pipefail

PORT="${1:-8000}"
export SGNS_DB_PATH="${SGNS_DB_PATH:-$(pwd)/var/sgns.db}"
export SGNS_ARTIFACT_DIR="${SGNS_ARTIFACT_DIR:-$(pwd)/var/artifacts}"

exec python3 -m uvicorn sgns.app:app --host 127.0.0.1 --port "${PORT}"
