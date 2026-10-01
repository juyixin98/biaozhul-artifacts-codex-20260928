# Lasso Path Backend

FastAPI backend for a Lasso regularization path. The production solver uses
cyclic coordinate descent with warm starts and does not use scikit-learn.
Scikit-learn is present only in tests and `scripts/smoke_validate.py` as an
independent mature-library oracle.

## Modules

- `models.py`: request/response contracts and explicit lambda and matrix limits.
- `parsing.py`: dense rows or COO triplets to NumPy/SciPy arrays. Duplicate
  sparse coordinates, malformed dimensions, and oversized payloads are rejected.
- `preprocessing.py`: frozen response centering and feature standardization.
  Sparse X stays in CSC; centered sparse columns are represented implicitly via
  residual offsets rather than materializing the full matrix.
- `solver.py`: soft-threshold cyclic coordinate descent, warm-started path,
  default lambda grid, KKT residual, duality gap, primal objective, support
  masks, and non-convergence reporting.
- `diagnostics.py`: request IDs, stage events, accept/reject/undetermined
  reasons, and aggregate-only matrix metadata. Raw rows are never logged.
- `reference.py`: optional scikit-learn comparison helper for verification.
- `app.py`, `service.py`: HTTP boundary and orchestration.
- `tests/`: independent assertions for numerical results, failure classes,
  sparse behavior, and API contracts.

## Model

The estimator minimizes

```text
0.5 * mean((y - intercept - Z @ beta)^2) + lambda * ||beta||_1
```

where `Z[:, j] = (X[:, j] - mean(X[:, j])) / std(X[:, j], ddof=1)`. The
intercept is unpenalized. Coefficients returned by the API are restored to the
original feature units; intercepts use the frozen preprocessing means.

The coordinate update is

```text
beta_j = soft_threshold(Z_j.T @ partial / n, lambda) / (Z_j.T @ Z_j / n)
```

where `partial` adds the old coordinate contribution back before thresholding.
Acceptance is based on the fixed-point/KKT residual: active coordinates satisfy
`g_j = lambda * sign(beta_j)`, inactive coordinates satisfy `|g_j| <= lambda`,
where `g = Z.T @ residual / n`. The duality gap is reported as supporting
evidence. Reaching `max_iter` does not raise away the result: the point is
returned with HTTP 200, `decision: "undetermined"`, status
`max_iterations`, iterations, KKT violation, and reason.

## Assumptions And Limits

- Numeric matrices and response only; NaN and infinite values are rejected.
- Constant or near-zero-variance columns are rejected by default, with column
  indices in the failure message. `zero_variance_policy="zero_fill"` is
  available for controlled local use.
- Sparse support is sparse-safe: the core path does not call `toarray()` on X.
  A test installs an exploding sparse matrix subclass to enforce this.
- Extremely collinear columns are valid but can make cyclic coordinate descent
  slow. This is reported as non-convergence when the KKT tolerance is not met.
- Default request limits are 10,000 rows, 2,000 features, and 2,000,000 sparse
  entries. Override them through `LASSO_MAX_ROWS`, `LASSO_MAX_FEATURES`, and
  `LASSO_MAX_NNZ`.

## Local Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-test.txt
```

## Verification

Run all independent tests:

```bash
.venv/bin/python -m pytest -q
```

Expected result: `20 passed` in this checkout. One Starlette/httpx
deprecation warning may be visible from `TestClient`; it does not affect the
numerical tests.

Run the synthetic oracle script:

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_validate.py
```

Expected judgment:

- `"accepted": true`
- every point has `"converged": true`
- `"kkt_violation"` is at most `1e-9`
- `"max_reference_error"` is below `2e-7`
- the smallest-lambda support is `[0, 2, 4]`

Start the HTTP service:

```bash
PYTHONPATH=src .venv/bin/uvicorn lasso_backend.app:app --host 127.0.0.1 --port 8000
```

Then check:

```bash
curl -s http://127.0.0.1:8000/health
```

The interactive contract is at `http://127.0.0.1:8000/docs`.

Example request skeleton:

```json
{
  "x": {"format": "coo", "n_rows": 3, "n_cols": 2,
         "rows": [0, 1, 2], "cols": [0, 1, 0], "data": [1.0, 2.0, 0.5]},
  "y": [1.0, 2.0, 3.0],
  "lambdas": [0.5, 0.1],
  "lambda_order": "descending",
  "warm_start": true,
  "max_iter": 10000,
  "tol": 1e-8,
  "zero_variance_policy": "reject"
}
```

## Verified Versions

The implementation was validated locally with Python 3.12.3 and:

- NumPy 2.5.3
- SciPy 1.18.1
- FastAPI 0.142.1
- Pydantic 2.13.5
- Uvicorn 0.54.0
- pytest 9.1.1
- httpx 0.28.1
- scikit-learn 1.9.1, verification oracle only
