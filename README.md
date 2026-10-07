# Restricted Python Mutation Testing Backend

A multi-module backend that mutation-tests restricted Python source via
AST rewriting, executes every mutant in an isolated copy of the project,
persists evidence to SQLite, and exposes results through a FastAPI
report interface.

## Modules

| Path | Responsibility |
| --- | --- |
| `app/models.py` | Pydantic input/output models; outcome and status enums |
| `app/mutator.py` | AST mutation engine (comparison / arithmetic / condition negation) |
| `app/runner.py` | Execution kernel: isolated copies, baseline, per-mutant pytest runs |
| `app/store.py` | SQLite evidence store (runs + mutant records) |
| `app/report.py` | FastAPI report interface with request identity |
| `app/config.py` | Settings (env-overridable), operator-set version |
| `fixtures/` | Synthetic algorithm packages with known mutant fates |
| `tests/` | Independent unit/integration/API tests with hand-written references |
| `scripts/verify.py` | End-to-end verification script |

## Mutation operators (operator set `ops-v1`)

- **comparison**: one mutant per alternative operator (`<` -> `<=`, `>`; `>=` -> `>`, `<=`; `==` <-> `!=`; ...)
- **arithmetic**: `+` <-> `-`, `*` -> `/`, `/` -> `*`, `//` -> `/`, `%` -> `*`
- **condition_negation**: `if`/`while` tests wrapped in `not (...)`

## Boundary semantics

Mutant fates (`MutantOutcome`):

- `invalid_syntax`: the mutated source does not compile; detected before
  any test execution.
- `startup_failure`: pytest exited with a code other than 0/1 (collection
  or import crash, e.g. a mutant that breaks module import).
- `killed`: the test suite failed (pytest exit 1) on the mutant.
- `survived`: the test suite passed on the mutant.
- `timeout`: the mutant's own test run exceeded `timeout_seconds`.
- `undetermined`: the mutant was detected as statically equivalent
  (mutation inside unreachable code) and was **not executed**. Equivalent
  mutants are never reported as "insufficient tests".

Run-level statuses (`RunStatus`) are distinct from mutant fates:

- `baseline_failed`: the unmutated suite is already red; no mutant verdicts.
- `baseline_timeout`: the unmutated suite exceeded
  `baseline_timeout_seconds`. This is deliberately separate from a
  per-mutant `timeout`.
- `invalid_source`: an original source file does not compile.
- `completed`: baseline green; every mutant has a fate.

Isolation: each run copies the project into a fresh temporary directory
(`tempfile.mkdtemp`), applies one mutation at a time inside the copy, and
restores the file afterwards. A SHA-256 digest of the original tree is
taken before and after the run; a mismatch is recorded as a warning in
the run summary and logged as an error.

Equivalence detection is intentionally conservative: only mutations in
statically unreachable code (statements after `return`/`raise`/
`continue`/`break` in the same block, and bodies of `if/while False:`)
are marked `undetermined`. Other undecidable equivalents run normally and
may survive; that is reported as `survived`, not `undetermined`.

Observability: logs carry `run_id` and `requester`; every HTTP response
echoes `X-Request-ID`; run summaries record the backend version,
operator-set version and Python version; `GET /runs/{id}/report`
separates failures (`invalid_syntax`/`startup_failure`/`timeout`) and
`undetermined` mutants from survivors.

## Running

```bash
python3 -m pytest tests/ -q          # full independent test suite
python3 scripts/verify.py            # end-to-end verification
uvicorn app.report:app --port 8000   # serve the API
```

API endpoints:

- `POST /runs/sync` - synchronous mutation run (body: `RunRequest`)
- `POST /runs` - background run (202)
- `GET /runs`, `GET /runs/{id}`, `GET /runs/{id}/mutants`,
  `GET /runs/{id}/report`
- `GET /health`

## Fixtures

- `fixtures/sample_pkg`: algorithm package whose 22 mutants have
  hand-derived fates (16 killed, 2 survived, 4 undetermined).
- `fixtures/timeout_pkg`: one mutant loops forever (per-mutant timeout).
- `fixtures/startup_pkg`: one mutant breaks module import at collection.
- `fixtures/baseline_fail_pkg`: red baseline.
- `fixtures/slow_baseline_pkg`: baseline exceeds its timeout.
- `fixtures/invalid_pkg`: original source with a syntax error.

## Known limitations / not implemented

- Background `POST /runs` returns a placeholder; poll `GET /runs` for the
  stored record (no per-run status endpoint for in-flight runs).
- Equivalence detection covers unreachable code only; general equivalent
  mutants are out of scope by design.
- Only single-comparator comparisons and the listed binary operators are
  mutated; boolean operators, augmented assignment and unary arithmetic
  are not mutated.
- Mutation of the test suite itself is excluded by convention
  (`test_dir` prefix), not by content analysis.
