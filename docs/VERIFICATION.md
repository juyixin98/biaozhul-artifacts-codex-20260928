# Verification Record

Date: 2026-09-30

## Environment

- Python: 3.12.3
- NumPy: 2.5.3
- SciPy: 1.18.1
- FastAPI: 0.141.1
- Uvicorn: 0.54.0
- Pydantic: 2.13.5
- HTTPX: 0.28.1
- pytest: 9.1.1

No virtual environment was created and no packages were installed.

## Automated Tests

Command:

```bash
python3 -m pytest -q
```

Result:

```text
24 passed, 1 warning in 2.45s
```

The single warning is Starlette's current deprecation notice for using
`httpx` with `TestClient`; it does not affect the tested API contract.

Compilation check:

```bash
python3 -m compileall -q entropic_ot tests
```

Result: exit code 0.

## Live HTTP Check

Port 8000 was already occupied, so the service was started on port 8001:

```bash
ENTROPIC_OT_LOG_DIR=manual_runs \
  python3 -m uvicorn entropic_ot.service:app --host 127.0.0.1 --port 8001
```

Request from `docs/api-example.json` returned HTTP 200 with:

- run ID: `20260930T052048610412Z-7e9e71c4`
- plan:
  `[[0.36552928931500245, 0.13447071068499758], [0.13447071068499758, 0.36552928931500245]]`
- original cost: `0.26894142136999516`
- regularized cost: `0.37988549304172253`
- entropy penalty: `0.11094407167172737`
- `converged=true`, `feasible=true`

A one-iteration non-uniform, high-cost, `epsilon=1e-8` request returned HTTP
509 with:

- run ID: `20260930T052055759099Z-8ed406ee`
- category: `resource_exhausted`
- best-effort `converged=false`, `feasible=false`
- exact row marginals, column error `0.3`

The corresponding replay logs are retained in `manual_runs/`.
