#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="$ROOT/fixtures/sample_algorithm_pkg"

echo "[1/2] Fixture baseline"
(cd "$FIXTURE" && python3 -m pytest -q -p no:cacheprovider)
baseline_status=$?
if [[ $baseline_status -ne 0 ]]; then
  echo "FAIL: fixture baseline tests did not pass" >&2
  exit "$baseline_status"
fi

echo "[2/2] Independent backend tests"
(cd "$ROOT" && python3 -m pytest -q -p no:cacheprovider)
backend_status=$?
if [[ $backend_status -ne 0 ]]; then
  echo "FAIL: independent backend tests did not pass" >&2
  exit "$backend_status"
fi

echo "PASS: fixture baseline and independent backend tests passed"
