#!/usr/bin/env bash
# Start the three local services used by the demo.
# Requires no accounts and no network: all state is in-process memory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "reference (correct) SUT : http://127.0.0.1:8000"
echo "buggy (planted bugs) SUT: http://127.0.0.1:8100"
echo "model-driven test API   : http://127.0.0.1:8200"

python3 -m uvicorn mdbt.services.reference:app --host 127.0.0.1 --port 8000 \
    --log-level warning &
PID_REF=$!
python3 -m uvicorn mdbt.services.buggy:app --host 127.0.0.1 --port 8100 \
    --log-level warning &
PID_BUG=$!
python3 -m uvicorn mdbt.api:app --host 127.0.0.1 --port 8200 \
    --log-level warning &
PID_API=$!

cleanup() {
    kill "$PID_REF" "$PID_BUG" "$PID_API" 2>/dev/null || true
}
trap cleanup EXIT

wait
