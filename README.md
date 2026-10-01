# Local Beta-Binomial Batch Update And Prediction Backend

This service maintains a Beta-Binomial model from success/failure batch
counts. It uses only local synthetic fixtures and a SQLite file.

## Statistical Assumptions

- Observations in each batch are Bernoulli trials with a shared probability
  `theta`.
- The prior is `theta ~ Beta(alpha, beta)` where both parameters are finite,
  strictly positive, and no larger than the local supported bound of `1e100`.
- Success and failure counts are finite, non-negative Python/JSON integers.
  Booleans, fractions, negatives, `NaN`, and infinity are rejected.
- Counts are sufficient statistics for the conjugate posterior:
  `theta | data ~ Beta(alpha + successes, beta + failures)`.
- For a future horizon of `n` trials, predictions are Beta-Binomial:
  `k | n,data ~ BetaBinomial(n, posterior_alpha, posterior_beta)`.
- The Bayesian posterior interval is explicitly called a credible interval.
  The separate data-only Wilson interval is explicitly called a frequentist
  confidence interval. They are not interchangeable.

Numerical probabilities are computed from log-scale rising products and a
log-sum-exp normalizer. This avoids direct binomial-coefficient overflow and
loss of cancellation for very large posterior parameters.

## Module Layout

- `app/models/contracts.py`: request/response contracts and strict Pydantic
  validation.
- `app/stats/kernel.py`: posterior, credible interval, Wilson interval, and
  stable Beta-Binomial prediction kernel.
- `app/persistence/repository.py`: SQLite schema, transaction handling, and
  immutable model/event rows plus mutable batch lifecycle status.
- `app/services/beta_service.py`: state transitions, current sufficient
  totals, revision/revert/restore semantics, posterior summaries, and
  predictions.
- `app/api/`: FastAPI dependency wiring and HTTP routes.
- `app/errors.py`: domain error categories and the shared JSON error
  contract.
- `experiments/oracle.py`: independently written reference formulas. It does
  not import the production statistics kernel.
- `experiments/run_experiment.py`: local reproducible scenarios.
- `tests/`: independent assertions for statistics, state transitions, API
  contracts, persistence failures, and replay logs.

## Version And Batch Semantics

Model creation writes version `0`. Each state-changing operation appends an
immutable event.

A revision performs the required order in one SQLite transaction:

1. Mark the original active batch `superseded`.
2. Append a withdrawal event with negative old sufficient statistics.
3. Insert the replacement active batch with `revision_of_batch_id`.
4. Append an addition event with the corrected sufficient statistics.

Reverting an active batch marks it `reverted` and appends negative deltas.
Restoring a standalone reverted batch reactivates it and appends positive
deltas. Recovering a superseded original withdraws the active replacement
first, then reactivates the original.

Events include the event type, deltas, resulting totals, an intermediate state
object, and a human-readable `judgement` explaining the decision.

## Error Categories

The response shape is:

```json
{
  "error": {
    "code": "STATE_CONFLICT",
    "category": "state_conflict",
    "message": "...",
    "details": {},
    "request_id": "..."
  }
}
```

Categories:

- `invalid_input`, HTTP 422: bad prior, bad counts, schema/type mismatch.
- `not_found`, HTTP 404: unknown model or batch.
- `state_conflict`, HTTP 409: revising/reverting a non-active batch, or an
  invalid restore transition.
- `resource_exhausted`, HTTP 413: aggregate-count capacity or predictive
  response-size capacity exceeded; SQLite disk/space failures are also mapped
  to this category.
- `persistence_failed`, HTTP 500: non-capacity SQLite failures.
- `computation_failed`, HTTP 500: non-finite values or normalization failure.

## Local Verification

Run all tests:

```bash
pytest -q --run-id local-verification
```

Expected result: all collected tests pass. Replay artifacts are written to
`test_runs/local-verification/`:

- `events.jsonl`: per-test start/finish records and explicitly logged
  intermediate states and judgement text.
- `manifest.json`: run id, exit status, Python/platform details, dependency
  versions, source hashes, and replay command.

Run the independent reproduction scenarios:

```bash
python3 -m experiments.run_experiment --output experiment_runs/latest.json
```

Expected judgement: every scenario has finite, non-negative predictive mass
with a reported sum numerically equal to one within tolerance.

Start the HTTP server:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

A minimal manual check is:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/models \
  -H 'content-type: application/json' \
  -d '{"alpha":1.0,"beta":1.0,"reason":"manual"}'
```

Use the returned `id` to add counts at `/models/{id}/batches`, inspect
`/models/{id}/posterior`, and call `/models/{id}/predict`.

## Configuration

Defaults are suitable for local use:

- `BETA_BINOMIAL_DB_PATH`, default `data/beta_binomial.sqlite3`.
- `BETA_BINOMIAL_MAX_PREDICTIVE_TRIALS`, default `10000`.
- `BETA_BINOMIAL_MAX_OBSERVATION_COUNT`, default `10^15`.
- `BETA_BINOMIAL_MAX_PRIOR_PARAMETER`, default `1e100`.

## Not Implemented

- Authentication, multi-tenant authorization, and production deployment are
  intentionally absent because all participants are local synthetic fixtures.
- PostgreSQL is not used; SQLite is the required local durable store.
- The server does not stream very large predictive distributions. Requests
  above the explicit local response limit return `RESOURCE_EXHAUSTED`.

