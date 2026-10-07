# Delta Debugging Service

A controlled delta debugging service: given a local test program and a
structured input that makes it fail with a *target* error, the service
minimizes the input (whole-record removal, ddmin) while the same target
error keeps reproducing.

Everything runs locally with synthetic fixtures. No production accounts
or real business data are involved.

## Module Layout

| Path | Responsibility |
| --- | --- |
| `ddmin_service/models.py` | Input model (`StructuredInput` with declared syntax constraints), verdicts, job spec |
| `ddmin_service/classifier.py` | Failure classification: `target` / `other` / `pass` / `indeterminate`, each with a reason |
| `ddmin_service/oracle.py` | Execution kernel: runs the program on a candidate, caches verdicts keyed by digest(program + env + input) |
| `ddmin_service/reducer.py` | ddmin over the record-removal operator, checkpointable state, 1-minimality verification |
| `ddmin_service/store.py` | SQLite evidence store: jobs, query cache, diagnostic events |
| `ddmin_service/service.py` | Job orchestration: create / run / resume / report |
| `ddmin_service/api.py` | FastAPI report and control interface |
| `ddmin_service/config.py` | Settings via env vars (`DD_DB_PATH`, `DD_TIMEOUT_SECONDS`, `DD_LIVE_QUERY_BUDGET`) |
| `fixtures/programs/` | Synthetic controlled programs (multi-condition crash, hang) |
| `fixtures/inputs/` | Sample structured inputs |
| `tests/` | Independent unit + integration tests (reference answers are hand-computed) |

## Behavior Conventions

1. **Failure categories.** Every candidate execution is classified as
   `target` (configured exit code, optionally plus a stderr pattern),
   `other` (any other non-zero exit), `pass` (exit 0), or
   `indeterminate` (timeout / no decision possible). The reason is
   recorded with each verdict.
2. **Syntax constraints.** Inputs are `{"records": [...]}` where each
   record is an object with a unique string `id`. The only reduction
   operator is whole-record removal, so every candidate keeps the
   declared constraints by construction.
3. **Cache binding.** Cache keys are SHA-256 over the program digest
   (argv + file bytes), an environment fingerprint (interpreter version
   + declared env overrides) and the canonical candidate JSON. Changing
   any of them is a cache miss.
4. **Minimality scope.** Reports claim only *1-minimality under the
   record-removal operator*: removing any single remaining record no
   longer deterministically reproduces the target error. No global
   minimality is claimed.

## Quickstart

```bash
./run.sh                 # uvicorn on 127.0.0.1:8000 (DD_PORT to change)
```

Create and run a job with the sample fixture:

```bash
python3 - <<'EOF'
import json
spec = {
  "program": ["python3", "fixtures/programs/sum_threshold_crash.py"],
  "input": json.load(open("fixtures/inputs/overflow_input.json")),
  "classifier": {"target_exit_codes": [3], "pass_exit_codes": [0], "timeout_seconds": 2},
  "sensitive_fields": ["secret"],
}
open("/tmp/dd_job.json", "w").write(json.dumps(spec))
EOF

curl -s -X POST 127.0.0.1:8000/jobs -H 'content-type: application/json' -d @/tmp/dd_job.json
# -> {"job_id":"...","state":"pending","request_id":"..."}
curl -s -X POST 127.0.0.1:8000/jobs/<job_id>/run
curl -s 127.0.0.1:8000/jobs/<job_id>/report
curl -s 127.0.0.1:8000/jobs/<job_id>/events
```

Expected conclusion: the report has `state=done`, `verdict=target`,
`minimized_record_ids=["m1","a1","a2"]` (the hand-computed minimal
trigger: mode=1 plus adds 70+30=100), `one_minimal=true`, and a
`minimality_scope` limited to the record-removal operator.

### Interrupt and resume

`POST /jobs/{id}/run?max_live_queries=N` pauses the job (`state=paused`)
after N live program executions; the checkpoint is persisted in SQLite.
Calling `run` again resumes from the checkpoint. Cached verdicts make
resumed runs deterministic and cheap.

### Diagnostics and masking

Every diagnostic event carries `job_id`, `request_id` (also returned as
the `x-request-id` response header), a timestamp and a detail payload
explaining why a candidate was accepted, rejected or undecidable.
Fields listed in `sensitive_fields` (default `secret`, `token`,
`password`) are persisted only masked as `***`.

## Testing

```bash
python3 -m pytest -q
```

Verified result at delivery: **22 passed**. Coverage includes:
classification of all four failure categories, ddmin against
hand-computed reference minima, 1-minimality verification, cache key
binding (program/env/input), end-to-end reduction through the HTTP API,
cache sharing across identical jobs, indeterminate and pass initial
inputs failing fast with reasons, event/request-id diagnostics, masking
of sensitive fields, and interrupt/resume reaching the same result as an
uninterrupted run.
