#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p test-results

export TEST_DATABASE_URL="${TEST_DATABASE_URL:-postgres:///opp300_test?host=/var/run/postgresql&sslmode=disable}"

echo "== unit tests =="
go test ./... 2>&1 | tee test-results/unit-tests.out

echo "== integration tests against PostgreSQL =="
go test -tags integration -v ./integration 2>&1 | tee test-results/integration-tests.out

echo "== go vet =="
go vet ./... 2>&1 | tee test-results/go-vet.out
go vet -tags integration ./... 2>&1 | tee -a test-results/go-vet.out

echo "saved: test-results/unit-tests.out test-results/integration-tests.out test-results/go-vet.out"
