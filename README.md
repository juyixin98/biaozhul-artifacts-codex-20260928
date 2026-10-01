# Irregular Lag Correlation Service

Python/FastAPI implementation for lag correlation of two irregularly sampled
numeric sequences. The project is intentionally split into statistical kernel,
API contract, SQLite persistence, diagnostics, reproducible experiments, and
independent tests.

## Statistical Contract

A lag shifts series B onto series A using:

```text
aligned_b_time = b_time - lag
```

A positive lag therefore means B is observed later than A.

For each requested lag:

1. Build every candidate edge whose absolute timing residual is within
   `tolerance`: `|b_time - lag - a_time| <= tolerance`.
2. Sort all candidate edges globally by `(absolute residual, A index, B index)`.
3. Greedily accept an edge only if neither endpoint has already been used.

That rule is explicitly one-to-one. When one side is more densely sampled, a
point on the sparse side cannot be repeated and accidentally weighted multiple
times. The dense side may have multiple candidates, but conflicts are resolved
using the globally closest residual.

The accepted pairs use fixed statistics:

- detrend: subtract the mean from the paired A and paired B values;
- standardization: population standard deviation z-score, `ddof=0`;
- statistic: Pearson correlation of the paired z-scores;
- result version: `irregular-lag-pearson-1.0.0`.

The minimum effective overlap is fixed to:

- at least 8 selected pairs; and
- pair fraction at least 0.5, where the denominator is
  `min(count(A), count(B))`.

The peak is the accepted lag with the largest absolute correlation. Ties use
the smallest lag. A correlation peak is only an alignment statistic. It is not
evidence of causation, causal direction, or a physical lagged effect.

## Status And Error Semantics

Per-lag statuses are part of the response:

- `accepted`: enough valid one-to-one pairs and both sides are non-constant.
- `unknown`: effective overlap is insufficient. The data do not support a
  correlation decision at that lag.
- `rejected`: enough pairs exist, but a constant selected A/B value set
  makes Pearson correlation undefined.

Failure codes are stable and explicit:

- `INSUFFICIENT_PAIR_COUNT`: fewer than 8 selected pairs.
- `INSUFFICIENT_PAIR_COVERAGE`: pair fraction below 0.5.
- `CONSTANT_SERIES`: one or both selected paired value sets are constant.
- `INVALID_SERIES`: HTTP 422 for malformed data, non-finite values, length
  mismatch, duplicate timestamps, invalid lag values, or non-positive
  tolerance.
- `DUPLICATE_REQUEST_ID`: HTTP 409 when a supplied `request_id` already exists.
- `ANALYSIS_NOT_FOUND`: HTTP 404 for an unknown stored identifier.

`unknown` is not represented by a fake zero correlation. The `correlation`
field is `null` for both `unknown` and `rejected`.

## Request Example

```json
{
  "request_id": "example-001",
  "a": {"times": [0.0, 1.0, 2.0], "values": [0.1, 0.5, 0.4]},
  "b": {"times": [0.5, 1.5, 2.5], "values": [0.1, 0.5, 0.4]},
  "lags": [0.5, 1.0],
  "tolerance": 0.25,
  "include_pairs": true
}
```

The response contains the result version, fixed config, per-lag correlation,
pair evidence, a detailed diagnosis object, peak summary, non-causal warning,
and status counts.

## Modules

- `lagcorr/kernel.py`: input validation, tolerance candidates, one-to-one
  matching, fixed detrending/standardization, per-lag status, and peak.
- `lagcorr/schemas.py`: FastAPI/Pydantic request and response contracts.
- `lagcorr/repository.py`: SQLite schema, transactional result storage, pair
  evidence persistence, and retrieval.
- `lagcorr/service.py`: application orchestration and response serialization.
- `lagcorr/api.py`: service routes and HTTP failure-category mapping.
- `lagcorr/logging_setup.py`: redacted logs with request ID and status counts.
- `lagcorr/experiments.py`: deterministic synthetic cases and independent
  brute-force/SciPy reference checks.
- `scripts/demo.py`: local end-to-end synthetic demo through the API app.
- `tests/`: independent concrete assertions for the kernel, API, storage,
  failure classes, pair details, and experiments.

Diagnostics contain structural key state: A/B counts, candidate edge count,
selected pair count, pair fractions, spans, largest selected gap, residual
statistics, and constant-side flags. Application logs never print sample
values. If measurements are sensitive, keep database files in an access-
controlled location.

## Run

Use the system Python directly as specified for this workspace:

```bash
python3 -m uvicorn lagcorr.api:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

The SQLite path defaults to `data/lagcorr.db` and can be changed with:

```bash
LAGCORR_DB_PATH=/tmp/lagcorr-demo.db python3 -m uvicorn lagcorr.api:app --port 8000
```

## Local Demo

```bash
python3 scripts/demo.py
```

The demo posts a synthetic known-offset sequence through the FastAPI app and
prints the peak, per-lag status, pair counts, and redacted diagnostic state.

## Reproduce Experiments

```bash
python3 -m lagcorr.experiments
```

The deterministic experiment set covers:

- known offset for uniform sampling;
- different sampling densities with close decoys;
- a long internal gap;
- constant sequences;
- insufficient overlap;
- an irregular/non-perfect case compared against an independent SciPy Pearson
  result.

Expected pair lists are generated by a separate brute-force oracle in
`lagcorr/experiments.py`; they are not copied from the production kernel.
Uniform special-case correlation values use `scipy.stats.pearsonr`.

## Run Tests

```bash
python3 -m pytest -q
```

Tests assert concrete correlations, exact selected indices, endpoint
uniqueness, candidate counts, failure categories, stored pair evidence,
duplicate identifiers, and long-gap/constant/insufficient-overlap behavior.
