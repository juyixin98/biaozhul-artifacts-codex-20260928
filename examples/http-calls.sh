#!/usr/bin/env bash
set -euo pipefail

BASE="${COORDINATOR_BASE_URL:-http://127.0.0.1:18080}"
SUBMIT_FILE="${1:-examples/submit-job.json}"

job_id="$(curl -sSf -X POST "$BASE/v1/jobs" \
  -H 'Content-Type: application/json' \
  --data-binary "@$SUBMIT_FILE" | jq -r .job.id)"

echo "job_id=$job_id"
echo "polling $BASE/v1/jobs/$job_id/progress"

for _ in $(seq 1 120); do
  state="$(curl -sSf "$BASE/v1/jobs/$job_id" | jq -r .state)"
  echo "state=$state"
  if [[ "$state" == "succeeded" || "$state" == "failed" ]]; then
    break
  fi
  sleep 0.25
done

echo "--- progress"
curl -sSf "$BASE/v1/jobs/$job_id/progress" | jq .
echo "--- attempts"
curl -sSf "$BASE/v1/jobs/$job_id/attempts" | jq .
echo "--- replay events"
curl -sSf "$BASE/v1/jobs/$job_id/events?limit=1000" | jq .
echo "--- merged result"
curl -sSf "$BASE/v1/jobs/$job_id/result" | jq .
