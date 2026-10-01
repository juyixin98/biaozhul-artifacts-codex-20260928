# Distributed Mean, Variance and Covariance Aggregation

This is a self-contained Go backend for aggregating partition-local mean, variance
and covariance summaries. It uses only the Go standard library for HTTP and
PostgreSQL for durable task state and replay.

## Versions Used

- Go: 1.23.4
- PostgreSQL client/server verified locally: 16.15
- PostgreSQL driver: `github.com/jackc/pgx/v5 v5.7.2`
- Transport: `net/http` with the Go 1.22+ method-aware `ServeMux`

## Layout

- `internal/protocol`: JSON request/response, state and error contracts.
- `internal/stats`: weighted Welford/West online accumulator and Chan merge.
- `internal/postgres`: migrations, transactional state storage and event log.
- `internal/service`: input validation, job/shard orchestration and error classes.
- `internal/httpapi`: standard-library HTTP endpoints.
- `internal/integration`: real HTTP + real PostgreSQL tests.
- `migrations`: SQL schema in the isolated `diststats_b` schema.

## Statistical Contract

Input vectors use a row-major flat array. For dimension `d`,
`len(values) == rows*d`; values are ordered as row 0 coordinates, row 1
coordinates, and so on.

Weights are non-negative finite frequency weights:

- omitted weights mean every observation has weight `1`;
- `w > 0` counts as one observation with frequency `w`;
- `w = 0` counts as one observation but contributes no mass;
- negative, NaN, infinite or non-numeric weights are rejected.

The stored mergeable state is:

- `count`: number of observed rows, including zero-weight rows;
- `weight`: aggregate frequency weight;
- `mean[]`;
- `m2[]`: weighted sum of squared deviations, not a raw sum of squares;
- row-major symmetric `covariance[]`.

Population statistics divide `m2` or covariance by `weight`. Sample statistics
use Bessel correction and divide by `weight - 1`. Fractional frequency weights
are allowed; the sample denominator follows the same aggregate weight
convention. When a statistic is undefined, it is emitted as JSON `null`:

- empty shard or total weight `0`: mean, variance and covariance are `null`;
- one effective observation (`weight == 1`): population variance is `0`,
  sample variance and sample covariance are `null`.

The online update is the weighted Welford/West recurrence. Partition summaries
are combined with Chan's parallel formula:

```text
delta = mean_b - mean_a
factor = weight_a * weight_b / (weight_a + weight_b)
m2 = m2_a + m2_b + delta^2 * factor
```

The covariance matrix uses the multivariate form with `delta_i * delta_j`.
The implementation never subtracts two large raw sums of squares.

## Numeric Acceptance Tolerance

The high-precision oracle in `internal/stats/reference_test.go` converts each
input float64 to an exact `math/big.Rat`, computes mean/variance/covariance
rationally, and only converts the final rational answer back to float64. It does
not call the production merge implementation.

Tests log the observed error for every mean, variance and covariance entry.
The acceptance tolerances are:

- mean: relative error `4e-12`, balanced-tree tests `2e-11`;
- variance/covariance: `max(1e-7 absolute, 4e-12 relative)`.

The absolute floor is deliberate: cancellation at offsets around `1e9` can
leave a few units in the last decimal place of a small variance, but errors
remain around `1e-8` rather than the catastrophic loss caused by raw square-sum
subtraction.

Covered cases include:

- `1e12` offset with tiny variance;
- bivariate correlated vectors around `1e9`;
- a one-row shard merged with a 501-row shard;
- weights from `1` to `1e9`;
- sequential, left-skewed, right-skewed and balanced merge trees.

## State and Replay Semantics

PostgreSQL rows live in `diststats_b`, isolated from unrelated local tables.

Submission runs in one transaction:

1. lock the job row `FOR UPDATE`;
2. reject an existing `(job_id, shard_id)` as a state conflict;
3. reject completed or capacity-full jobs;
4. merge the shard's stable summary into current state;
5. insert the shard, event and updated job state.

Each replay event contains:

- run ID and job ID;
- monotonic sequence;
- shard ID;
- event type;
- error class and machine-readable reason/code;
- before/after state views;
- accepted request payload or parseable rejected request JSON.

Duplicate shard IDs are idempotency keys for that job: retries do not merge a
second time. Different dimensions cannot merge because each shard is built with
the job's declared dimension and malformed row-major values are rejected before
state access.

## API

### Create a run

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "run_id": "run-20260929-demo",
    "dimension": 2,
    "estimator": "population",
    "expected_shards": 2
  }'
```

`estimator` is `population` or `sample`; omitted means `population`.

### Submit a partition

Use the `id` returned by create, for example `job_...`:

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/jobs/job_.../shards \
  -H 'Content-Type: application/json' \
  -d '{
    "shard_id": "small",
    "producer_id": "worker-2",
    "values": [4, 8]
  }'

curl -sS -X POST http://127.0.0.1:8080/v1/jobs/job_.../shards \
  -H 'Content-Type: application/json' \
  -d '{
    "shard_id": "large",
    "producer_id": "worker-1",
    "values": [0, 0, 2, 4]
  }'
```

For these rows, the population result is:

- weight `3`;
- mean `[2, 4]`;
- variance `[8/3, 32/3]`;
- covariance matrix:

```json
[2.6666666666666665, 5.333333333333333, 5.333333333333333, 10.666666666666666]
```

### Read the current result

```bash
curl -sS http://127.0.0.1:8080/v1/jobs/job_...
```

### Replay decisions and intermediate state

```bash
curl -sS http://127.0.0.1:8080/v1/jobs/job_.../replay | jq .
```

### Health

```bash
curl -sS http://127.0.0.1:8080/healthz
```

## Error Categories

Errors have a stable `error_class` and `code`:

- `input_error`, HTTP 400: invalid JSON, missing ID, wrong shape, bad weights;
- `state_conflict`, HTTP 409: duplicate run or shard, job already complete;
- `not_found`, HTTP 404: unknown job;
- `resource_exhausted`, HTTP 413 for request bodies over 4 MiB, HTTP 507 for
  PostgreSQL resource-limit errors;
- `computation_failure`, HTTP 422: non-finite statistic or weight overflow;
- `storage_failure`, HTTP 500: unexpected database encoding/IO failure.

## Run Locally

On a fresh PostgreSQL installation where your user can create a database:

```bash
createdb diststats
export DATABASE_URL='postgres:///diststats?sslmode=disable'
```

On this prepared local machine, an existing `opp298` database is available to
the `admin` role. The application creates its own `diststats_b` schema, so it
does not touch unrelated tables:

```bash
export DATABASE_URL='postgres:///opp298?sslmode=disable'
```

Start:

```bash
go mod download
HTTP_ADDR=':8080' \
DATABASE_URL="$DATABASE_URL" \
MIGRATIONS_DIR='migrations' \
PG_MAX_CONNS='8' \
go run ./cmd/server
```

Migrations run automatically at startup.

## Test Commands

Numeric tests do not require PostgreSQL:

```bash
go test ./internal/stats -v
```

Full tests use the local PostgreSQL database. By default they connect to
`postgres:///opp298_test?sslmode=disable`; override with `TEST_DATABASE_URL`:

```bash
TEST_DATABASE_URL='postgres:///opp298_test?sslmode=disable' go test ./... -v
```

The integration test truncates only `diststats_b.jobs` and cascades to the
application-owned shard/event tables.

## Verified Local Results

The following commands were executed successfully on 2026-09-29 in Asia/Shanghai:

```text
go test ./...
?   diststats/cmd/server             [no test files]
?   diststats/internal/httpapi       [no test files]
?   diststats/internal/postgres      [no test files]
?   diststats/internal/protocol      [no test files]
?   diststats/internal/service       [no test files]
ok  diststats/internal/integration   0.570s
ok  diststats/internal/stats         (cached)
```

A live run with `run_id=run-20260929-demo-002` accepted shards `small` and
`large`, returned status `complete`, and replay recorded three events: two
acceptance events with before/after state and one `DUPLICATE_SHARD` rejection.

