APP=bin/stream-checkpoint-lab

.PHONY: build test integration race run postgres-run clean

build:
	go build -o $(APP) ./cmd/server

test:
	go test ./... -count=1 -timeout 10s

integration:
	go test -tags integration ./... -count=1 -timeout 20s

race:
	go test -race ./... -count=1 -timeout 20s

run:
	go run ./cmd/server -config configs/server.json

postgres-run:
	go run ./cmd/server -config configs/server-postgres.json

clean:
	rm -rf bin

