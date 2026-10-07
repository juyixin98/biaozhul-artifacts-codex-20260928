# Reproduction Guide

Everything runs locally. No accounts, no business data, no network access is
needed (the test SUTs keep all state in memory).

## 1. Environment

Python 3.12 with the globally installed packages:
`fastapi`, `uvicorn`, `pydantic`, `httpx`, `pytest`.
Exact versions used are pinned in [requirements.lock](../requirements.lock).

Do not create a virtualenv or install packages; import the global ones.

## 2. Run the independent test suite

```bash
python3 -m pytest tests -q
```

Expected: `29 passed`.

The tests assert concrete outcomes and failure categories:

* input-model validation rejects malformed/non-deterministic/unreachable
  models as `input_error`;
* the oracle predicts without mutating state and tracks resources
  independently (interleaving test);
* the reference service accepts the happy path and rejects all three
  planted-bug histories with exact HTTP 409 states;
* a buggy campaign finds exactly three `state_conflict` findings, with
  minimised lengths 6 / 4 / 5 and generated replay scripts;
* reduction keeps required preconditions (deleting `deliver` is rejected via
  revalidation, never assumed);
* the same seed reproduces identical summaries;
* `input_error`, `state_conflict`, `resource_exhausted` (dead SUT, step
  budget) and `computation_failure` (non-JSON / bad envelope) are distinct;
* the report API contract (validate, run, steps, findings, replay, 404/422).

## 3. Generate the checked-in demo results

```bash
python3 scripts/generate_results.py
```

This runs both local SUTs through the full campaign (seed `20261007`,
17 guided + 40 fuzz steps each), writes:

* `artifacts/demo_evidence.db` - SQLite evidence (runs/steps/findings);
* `artifacts/replays/replay_run-demo-buggy-*.py` - three minimal replay
  scripts (stdlib only);
* [docs/RESULTS.md](RESULTS.md) - human-readable summary.

Expected last lines:

```text
reference findings: 0
buggy findings: 3
```

## 4. Replay a detected bug against real HTTP services

Terminal 1/2/3:

```bash
python3 -m uvicorn mdbt.services.reference:app --host 127.0.0.1 --port 8000
python3 -m uvicorn mdbt.services.buggy:app     --host 127.0.0.1 --port 8100
```

Then (the scripts bypass ambient proxies and use unique ids):

```bash
python3 artifacts/replays/replay_run-demo-buggy-<suffix>_f00.py \
    --base-url http://127.0.0.1:8100            # prints BUG REPRODUCED
python3 artifacts/replays/replay_run-demo-buggy-<suffix>_f00.py \
    --base-url http://127.0.0.1:8000 --expect-clean
```

Finding map: `f00` repeated close, `f01` out-of-order submit,
`f02` cancel after delivery.

## 5. Use the report/control API

```bash
python3 -m uvicorn mdbt.api:app --host 127.0.0.1 --port 8200
bash scripts/call_api_examples.sh
```

Or start everything at once with `bash scripts/run_demo.sh`.

Key endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/models/validate` | validate a model document |
| POST | `/runs` | run a campaign (`sut`: `reference`/`buggy`/`http`) |
| GET | `/runs` | list runs |
| GET | `/runs/{id}` | run header, seed/config, summary, error |
| GET | `/runs/{id}/steps` | full per-step evidence |
| GET | `/runs/{id}/findings` | findings + reduction metadata |
| GET | `/runs/{id}/findings/{i}/replay` | minimal replay script |

## 6. Error category contract

| Category | Meaning | HTTP mapping |
|---|---|---|
| `input_error` | malformed model/command, unknown action/resource | 400/404/422 |
| `state_conflict` | well-formed op illegal in current state | 409 |
| `resource_exhausted` | step/connect/timeout budget exhausted | 429/503, dead SUT |
| `computation_failure` | internal/SUT wire-contract failure | 500/502, bad body |

## 7. How the four required properties are enforced

1. **Model transition vs SUT call separation** - `FSMModel.expected()` is
   pure (no HTTP import), `Executor.step()` predicts, then calls the SUT
   once, then advances state via `apply_observation()`.
2. **Seed + all scheduling choices recorded** - every step stores the seed,
   RNG state before/after, mode, candidate set, chosen command and
   rationale (see `steps[].decision`).
3. **Reduction preserves preconditions or revalidates** - each candidate
   deletion is replayed from a fresh SUT and accepted only when the same
   discrepancy signature (prediction, observed accept flag, category,
   action, model state, observed state) reproduces.
4. **Failure reports include a minimal replay script** - every finding
   stores a stdlib-only, executable replay script on disk and in SQLite.
