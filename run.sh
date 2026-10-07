#!/usr/bin/env bash
# Start the delta debugging service locally.
set -euo pipefail
cd "$(dirname "$0")"
exec uvicorn ddmin_service.api:app --host 127.0.0.1 --port "${DD_PORT:-8000}"
