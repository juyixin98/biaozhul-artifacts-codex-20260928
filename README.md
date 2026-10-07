# Delta Debugging Service

A controlled-test delta debugging service: given a structured input that
makes a target program fail with a specific error, it reduces the input
to a 1-minimal candidate (under the block-removal operator) that still
reproduces the same error.

## Layout

| Path | Responsibility |
| --- | --- |
| `app/models.py` | Input model: blocks, documents, declared syntax constraints, verdict enum, sensitive-value masking |
| `app/kernel.py` | Execution kernel: runs the target program, classifies outcome into target / other / pass / indeterminate, cache-key construction |
| `app/reducer.py` | Reduction kernel: greedy 1-minimization with checkpoint/resume, cache-aware evaluation |
| `app/store.py` | Evidence storage (SQLite): jobs, step log, run cache |
| `app/api.py` | Report interface (FastAPI): create/resume jobs, status, step evidence, final report |
| `app/config.py` | Configuration (env-overridable), environment fingerprint |
| `fixtures/target_program.py` | Local synthetic target program with a multi-condition trigger |
| `fixtures/sample_input.txt` | Sample failing input |
| `tests/` | Independent unit and integration tests |
| `scripts/smoke.py` | Smoke check against a running service |

## Behavior contract

1. **Failure classification** distinguishes `target` (the error being
   preserved), `other` (a different error), `pass`, and `indeterminate`
   (timeout / cannot decide).
2. **Syntax constraints** are declared in `Document.validate_syntax`
   (exactly one header first, one footer last, settings are `key=value`).
   The reduction operator only deletes setting blocks, so every candidate
   stays syntactically valid by construction.
3. **Cache keys** bind the target program digest, the environment
   fingerprint (Python version, OS, machine), and the candidate input.
4. **Minimality claim** is only "1-minimal under the block-removal
   operator"; no global-shortest claim is made.
5. Every decision is logged with job id, step number, action
   (verify / remove-accepted / remove-rejected), verdict, and a reason.
   Sensitive setting values (`password`, `secret`, `token`, `api_key`)
   are masked as `key=***` in all diagnostics and API output.

## The fixture

`fixtures/target_program.py` fails with the target error (exit 3) only
when ALL of `mode=strict`, `level>=3`, `debug=on` are present; malformed
input exits 2 (other error); `simulate_hang=1` hangs (indeterminate via
timeout); otherwise exit 0 (pass).

## Run the tests

```bash
python3 -m pytest -q
```

Verified result on this machine: **26 passed** (unit + integration,
including hand-written reference answer, independent 1-minimality check,
and interruption/resume convergence).

## Run the service

```bash
python3 -m uvicorn app.api:app --port 8000
```

Then, in another shell:

```bash
python3 scripts/smoke.py http://127.0.0.1:8000
```

Observed output:

```
create: 200 interrupted
resume: 200 done | 1-minimal under the block-removal operator only (no claim of global minimum)
report: BEGIN config-v1 | mode=strict | level=5 | debug=on | END config-v1
```

(If your shell exports a SOCKS proxy, clear `all_proxy`/`http_proxy` etc.
for the smoke client, or use `httpx[socks]`.)

## API

- `POST /jobs` — body `{"blocks": [...], "max_steps": null}`; `max_steps`
  limits reduction steps to simulate interruption. Returns job id and
  status (`done` / `interrupted`).
- `POST /jobs/{job_id}/resume` — resume an interrupted job from its
  checkpoint.
- `GET /jobs/{job_id}` — job status and masked current candidate.
- `GET /jobs/{job_id}/steps` — full evidence log with per-step verdicts
  and accept/reject reasons.
- `GET /jobs/{job_id}/report` — final masked candidate, size, and the
  scoped minimality claim.

## Configuration

Environment variables: `DDSVC_DB` (SQLite path), `DDSVC_TARGET` (target
program path), `DDSVC_TIMEOUT` (per-run timeout seconds, default 5).
