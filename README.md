# Exact Global Top-K Over Local Ordered Shards

This service computes exact global top-k candidates from independent local HTTP
shard participants backed by PostgreSQL. It is a runnable implementation, not a
mock interface: shards execute ordered SQL pages, the coordinator validates a
strict page protocol, and every accepted/rejected decision is persisted with a
request ID for replay.

## Ranking And Stopping Rule

The global order is a total order:

```text
score DESC, item_id ASC
```

`item_id` is globally unique inside a generation. A shard page carries:

- zero or more ordered candidates, at most `limit`;
- the cursor equal to the last returned key;
- `has_more=true` only after reading a real `limit+1` probe row;
- `upper_bound`, which is that probe row's exact key;
- an HMAC receipt covering generation, shard, cursor, and upper bound.

The coordinator keeps a size-k min heap ordered by the same global order. A
shard is stopped only when its known upper bound is not better than the current
kth result. Equal bounds do not displace an existing unique candidate, so the
inclusive tie boundary is safe. A timeout or protocol/bound failure returns the
partial result, known bounds, and `exact=false, complete=false`; it can never be
labelled exact.

A generation is an immutable data world. Every shard page must return the
requested generation, and coordinator query records are stored under that
generation. Re-seeding the same generation with different data is rejected by
checksum.

## Module Responsibilities

- `internal/domain`: total ordering, page protocol, failure classes.
- `internal/kernel`: heap merge, page validation, HMAC verification, early
  termination proof, structured decision events.
- `internal/storage`: PostgreSQL schema, immutable generations, ordered shard
  reads, query/result/status/event persistence, replay reads.
- `internal/participant`: standard-library HTTP shard service and client.
- `internal/service`: coordinator HTTP API, request IDs, failure-to-status
  mapping, storage-backed replay.
- `internal/fixture`: deterministic local dataset and independent expected
  answers.
- `cmd/topkd`: schema initialization, seeding, and service runtime.
- `configs`: local configuration.
- `scripts`: local database setup.

## Local Requirements

- Go 1.23.4
- PostgreSQL 16.x (validated with client/server 16.15 locally)
- github.com/jackc/pgx/v5 v5.7.6
- No external business accounts or data

The default config uses the Unix socket `/var/run/postgresql` as user `admin`
and database `opp297_topk`.

## Setup And Run

```bash
./scripts/setup-local-db.sh
go build -o bin/topkd ./cmd/topkd
./bin/topkd init-db configs/topk.local.json
./bin/topkd seed configs/topk.local.json
./bin/topkd run configs/topk.local.json
```

The coordinator listens on `http://127.0.0.1:18097`; four local shard services
listen on ports `18100` through `18103`.

Query:

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -d '{"generation":"gen-fixture-v1","k":6}' \
  http://127.0.0.1:18097/v1/top-k
```

Expected judgment:

- HTTP 200;
- `exact=true`, `complete=true`;
- exact IDs in order:
  `item-a00`, `item-a01`, `item-a02`, `item-b00`, `item-a03`, `item-b01`;
- threshold is `{score:99,id:"item-b01"}`;
- `stop_proof.total_rows_read=15` for page size 3 because the score-99 tie
  boundary must be resolved;
- shard 3 is empty and reports zero rows.

Use the returned `request_id` for replay:

```bash
REQ='<request_id>'
curl -sS "http://127.0.0.1:18097/v1/replay/$REQ"
```

Replay should include query status, per-shard state, results, every page event,
and a final `STOP_PROVED` event.

Fault demos use local test injection headers:

```bash
curl -sS -H 'X-Test-Fault: bad_bound:0' \
  -H 'Content-Type: application/json' \
  -d '{"generation":"gen-fixture-v1","k":1}' \
  http://127.0.0.1:18097/v1/top-k
```

Expected: HTTP 503, `exact=false`, `complete=false`,
`failure.class="BOUND_VIOLATION"`.

`X-Test-Fault: slow:2` produces `SHARD_TIMEOUT`; `wrong_generation:1` produces
`GENERATION_MISMATCH`. The shard runtime only honors these faults when local
fault injection is enabled in this development build/config.

## Tests

Fast unit tests, including an independent `sort.Slice` oracle over the original
fixture and explicit row counts:

```bash
go test ./...
```

PostgreSQL and real HTTP integration tests:

```bash
go test -tags integration ./internal/service -v
```

The integration suite expects database `opp297_topk` to already exist. Override
with:

```bash
TOPK_TEST_DATABASE_URL='postgres://user@/dbname?host=/var/run/postgresql&sslmode=disable' \
  go test -tags integration ./internal/service -v
```

Integration tests use unique generations and remove their generation and query
records after the test.

## Diagnostics And Sensitive Data

Logs and replay events contain request IDs, generation fingerprints, shard IDs,
cursors, upper bounds, thresholds, rows read, and failure classes/reasons. They
do not return `sensitive_label` from the API. Logs redact the generation string
to first/last fragments. Item IDs are synthetic local identifiers.
