#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DB_NAME="${SMOKE_DB_NAME:-opp300_smoke}"
COORD_PORT="${COORD_PORT:-19090}"
WORKER_PORT="${WORKER_PORT:-19091}"
ARTIFACT_DIR="${SMOKE_ARTIFACT_DIR:-./var/smoke-artifacts-script}"
RESULT_DIR="test-results/smoke"
RUN_ID="$(date +%s%N)"
ARTIFACT_DIR="${ARTIFACT_DIR}-${RUN_ID}"

mkdir -p "$RESULT_DIR" tmp
ADMIN_PSQL="${ADMIN_PSQL:-sudo -n -u postgres psql}"
DATABASE_URL="postgres:///${DB_NAME}?host=/var/run/postgresql&sslmode=disable"

cleanup() {
  if [[ -f tmp/smoke-coordinator.pid ]]; then
	kill "$(cat tmp/smoke-coordinator.pid)" 2>/dev/null || true
  fi
  if [[ -f tmp/smoke-worker.pid ]]; then
	kill "$(cat tmp/smoke-worker.pid)" 2>/dev/null || true
  fi
}
trap cleanup EXIT

$ADMIN_PSQL -v ON_ERROR_STOP=1 \
  -c "DROP DATABASE IF EXISTS ${DB_NAME};" \
  -c "CREATE DATABASE ${DB_NAME} OWNER admin;"

go build -o bin/coordinator ./cmd/coordinator
go build -o bin/worker ./cmd/worker

setsid env \
  DATABASE_URL="$DATABASE_URL" \
  HTTP_ADDR="127.0.0.1:${COORD_PORT}" \
  COORDINATOR_BASE_URL="http://127.0.0.1:${COORD_PORT}" \
  WORKER_URL="http://127.0.0.1:${WORKER_PORT}" \
  ARTIFACT_DIR="$ARTIFACT_DIR" \
  LEASE_MS=10000 \
  SPECULATE_AFTER_MS=150 \
  HEARTBEAT_INTERVAL_MS=50 \
  ./bin/coordinator > "$RESULT_DIR/coordinator.log" 2>&1 &
echo $! > tmp/smoke-coordinator.pid

setsid env WORKER_HTTP_ADDR="127.0.0.1:${WORKER_PORT}" \
  ./bin/worker > "$RESULT_DIR/worker.log" 2>&1 &
echo $! > tmp/smoke-worker.pid

for i in $(seq 1 50); do
  if curl -sf "http://127.0.0.1:${COORD_PORT}/healthz" >/dev/null \
    && curl -sf "http://127.0.0.1:${WORKER_PORT}/healthz" >/dev/null; then
	break
  fi
  sleep 0.1
done

run_case() {
  local name=$1
  local file=$2
  local expected=$3
  local request="$RESULT_DIR/${name}-request.json"

  jq --arg key "smoke-${name}-$(date +%s%N)" '.idempotency_key=$key' "$file" > "$request"
  local job_id
  job_id=$(curl -sf -X POST "http://127.0.0.1:${COORD_PORT}/v1/jobs" \
	-H 'Content-Type: application/json' --data-binary "@$request" | jq -r .job.id)

  local state=unknown
  for _ in $(seq 1 120); do
	state=$(curl -sf "http://127.0.0.1:${COORD_PORT}/v1/jobs/${job_id}" | jq -r .state)
	if [[ "$state" == "succeeded" || "$state" == "failed" ]]; then
	  break
	fi
	sleep 0.1
  done
  [[ "$state" == "$expected" ]]

  curl -sf "http://127.0.0.1:${COORD_PORT}/v1/jobs/${job_id}" > "$RESULT_DIR/${name}-job.json"
  curl -sf "http://127.0.0.1:${COORD_PORT}/v1/jobs/${job_id}/progress" > "$RESULT_DIR/${name}-progress.json"
  curl -sf "http://127.0.0.1:${COORD_PORT}/v1/jobs/${job_id}/attempts" > "$RESULT_DIR/${name}-attempts.json"
  curl -sf "http://127.0.0.1:${COORD_PORT}/v1/jobs/${job_id}/events?limit=1000" > "$RESULT_DIR/${name}-events.json"
  curl -sf "http://127.0.0.1:${COORD_PORT}/v1/jobs/${job_id}/result" > "$RESULT_DIR/${name}-result.json"
  jq -n --arg name "$name" --arg job "$job_id" --arg state "$state" \
	'{name:$name,job_id:$job,state:$state}' | tee "$RESULT_DIR/${name}-summary.json"
}

run_case abnormal examples/speculative-failure-job.json succeeded
run_case normal examples/submit-job.json succeeded

find "$ARTIFACT_DIR/staging" -type f | wc -l > "$RESULT_DIR/staging-file-count.txt"
find "$ARTIFACT_DIR/published" -type f | wc -l > "$RESULT_DIR/published-file-count.txt"

echo "smoke evidence saved to $RESULT_DIR"
