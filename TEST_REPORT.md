# Test Report

Environment captured on 2026-09-30, Asia/Shanghai:

```text
Python 3.12.3
fastapi==0.141.1
pydantic==2.13.5
numpy==2.5.3
scipy==1.18.1
pytest==9.1.1
httpx==0.28.1
uvicorn==0.54.0
```

Commands and observed result:

```bash
pytest -q
# 26 passed, 1 warning in 0.92s
```

The clean virtual-environment verification used
`/tmp/entropic-ot-venv` with the pinned `requirements.txt`. The remaining
warning comes from Starlette's bundled TestClient importing `httpx`; it does
not affect solver behavior.

Coverage summary:

- Hand-derived 2x2 interior solution asserts all four plan entries and both
  objective components independently of the solver.
- SciPy `linprog`/HiGHS provides an independent LP oracle for the small
  low-regularization comparison.
- The large-cost test uses `C=1e12`, `epsilon=1e-8`; underflowed off-diagonal
  entries become exact zeros without NaN or infinite values and preserve both
  marginals.
- `C=7000`, `epsilon=1` checks extreme but representable off-diagonal edges.
- Zero rows, zero columns, and all-zero masses return exact zeros and a clear
  stop reason.
- A deliberately tiny iteration budget returns
  `503 resource_exhausted`, `feasible=false`, and includes the final run
  number and sampled two-sided residual evidence.
- Injected non-finite dual arithmetic returns `computation_failed` rather
  than a partially satisfied transport plan.
- API tests verify model parse failure, semantic mass mismatch, successful
  JSONL replay evidence, and the resource-exhausted contract.
- A non-unit total-mass case verifies scale conservation and recomputes both
  objective components independently.
- A non-finite API tolerance is rejected at the model layer as an input
  error, and JSONL/error serialization replaces non-finite diagnostic floats
  with explicit null values.
