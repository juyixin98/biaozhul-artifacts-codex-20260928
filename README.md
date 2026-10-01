# Incremental Nondominated Sorting Backend

A small FastAPI service for exact finite-candidate multi-objective
non-dominated sorting. The same kernel supports initial ranking, incremental
insertion, and incremental deletion.

## Modules

- `app/parsing.py`: strict request parsing and specific validation error codes.
- `app/kernel.py`: exact dominance DAG, longest-path front numbers, insert and delete.
- `app/evidence.py`: independent all-pairs peeling reference and numerical diagnostics.
- `app/service.py`: thread-safe in-memory datasets.
- `app/api.py` and `app/main.py`: FastAPI routes, request IDs, logs and envelopes.
- `tests/`: independent unit and integration tests with concrete expected fronts.
- `fixtures/`: synthetic two-dimensional, high-dimensional/degenerate and invalid examples.

## Semantics

Each objective has a fixed direction, either `min` or `max`. Candidate `p`
dominates candidate `q` when `p` is no worse on every objective and strictly
better on at least one. Equal vectors do not dominate each other, but they
remain distinct candidate identities and each retains its own dominance edges
to other objects.

The order predicate uses exact declared numeric values. A positive
`tolerance` only creates a diagnostic near-equality warning. It is never
inserted into the ordering predicate because a loose predicate can create
non-transitive cycles.

The kernel ranks nodes by longest path in the exact dominance DAG. On insert,
new dominance edges can promote affected descendants. On delete, only the
deleted node's reachable descendants are topologically recomputed. Duplicate
deletion leaves the other duplicate object in place, so its retained
domination relationships can preserve lower fronts until it too is deleted.

## Run

Use the system Python as requested; no virtual environment is created:

```bash
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Alternatively:

```bash
./scripts/run_dev.sh
```

Open <http://127.0.0.1:8000/docs> for the interactive API.

## Examples

Stateless sort:

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-sort' \
  --data @fixtures/two_dimensional_mixed_directions.json \
  http://127.0.0.1:8000/api/v1/sort
```

Expected exact fronts for that fixture are:

```json
[["A", "D"], ["E"], ["C"], ["B"], ["F"]]
```

Create, insert and delete:

```bash
curl -sS -X POST \
  -H 'Content-Type: application/json' \
  --data @fixtures/two_dimensional_mixed_directions.json \
  http://127.0.0.1:8000/api/v1/datasets

curl -sS -X POST \
  -H 'Content-Type: application/json' \
  -d '{"candidates":[{"id":"P","values":{"cost":0,"reliability":3}}]}' \
  http://127.0.0.1:8000/api/v1/datasets/fixture-two-dimensional/candidates:batchInsert

curl -sS -X DELETE \
  http://127.0.0.1:8000/api/v1/datasets/fixture-two-dimensional/candidates/P
```

Missing objectives are rejected with HTTP 422 and a dedicated code:

```bash
curl -sS -X POST \
  -H 'Content-Type: application/json' \
  --data @fixtures/missing_objective_invalid.json \
  http://127.0.0.1:8000/api/v1/sort
```

The response error is `MISSING_OBJECTIVE` and lists the absent objective
names.

## Explainability

Every success response contains:

- `request_id`, also mirrored in the `X-Request-ID` response header.
- API and algorithm versions plus `kernel_version`.
- ordered `processing` entries with component location and action.
- candidate ranks and deterministic sorted layers.
- an `evidence` block containing comparison count, pairwise dominance edge count,
  duplicate groups, an independently generated peeling reference, and graph
  integrity flags.
- a `diagnostics` block for constant objectives, exact duplicate count and
  non-fatal near-equality uncertainties.

Errors use a separate top-level `error.code/message/details/location` object.
Diagnostic uncertainty is kept under `warnings` and never masks a request
failure.

## Tests

```bash
python3 -m pytest -q
```

The suite includes a concrete two-dimensional hand calculation, duplicate
deletion, mixed directions, a four-dimensional constant-objective case,
incremental insertion promotions, and randomized insert/delete sequences.
Random tests compare the kernel with an independently written all-pairs
peeling implementation in the test file rather than asking the kernel to
generate its own expected answer.
