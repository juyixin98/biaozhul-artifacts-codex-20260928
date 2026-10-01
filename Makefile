GO ?= go

.PHONY: tidy fmt test integration vet setup-db run-coordinator run-worker demo

fmt:
	$(GO) fmt ./...

tidy:
	$(GO) mod tidy

test:
	$(GO) test ./...

vet:
	$(GO) vet ./...
	$(GO) vet -tags integration ./...

integration:
	TEST_DATABASE_URL="$${TEST_DATABASE_URL:-postgres:///opp300_test?host=/var/run/postgresql&sslmode=disable}" \
	  $(GO) test -tags integration -v ./integration

setup-db:
	scripts/setup-db.sh

run-coordinator:
	$(GO) run ./cmd/coordinator

run-worker:
	$(GO) run ./cmd/worker

demo:
	examples/http-calls.sh examples/speculative-failure-job.json
