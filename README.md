# Local Skip-Gram Negative Sampling Backend

A small-corpus FastAPI service that trains deterministic Skip-Gram with Negative
Sampling (SGNS), stores model artifacts and replay logs in SQLite/local files,
and serves word vectors and similarity queries.

All fixtures are local and synthetic. The evaluation measures local
co-occurrence recovery only; it does not claim semantic understanding or
downstream language-performance improvement.

## Module Boundaries

- `app/config.py`: validated training hyperparameters.
- `app/data.py`: corpus validation, frozen vocabulary, counts/frequencies,
  token sub-sampling, and corpus fingerprint.
- `app/sampling.py`: symmetric pair generation and negative-sample policy.
- `app/model.py`: SGNS numeric step, input/output matrices, artifact save/load.
- `app/training.py`: deterministic RNG streams, training order, replay events.
- `app/storage.py`: SQLite runs/events/evaluations and artifact registration.
- `app/prediction.py`: vectors, cosine similarity, nearest neighbors.
- `app/evaluation.py`: synthetic co-occurrence ranking.
- `app/api.py` and `app/main.py`: HTTP contract and uvicorn entry point.
- `tests/`: independent math references and behavior/error tests.

## Fixed Training Semantics

- Vocabulary tokens must match `^(?![0-9_])\w+$`. OOV tokens are removed by
  `min_count` before a run and then the vocabulary is frozen.
- Vocabulary ordering is count descending, then token ascending. The same
  corpus and `min_count` therefore get identical token IDs.
- Frequencies are `count / total_count`. The negative distribution is
  `count^0.75 / sum(count^0.75)`.
- Sub-sampling uses the Word2Vec equivalent keep probability
  `min(1, (sqrt(f/t)+1) * t/f)` with one independent uniform draw per token
  occurrence.
- Symmetric skip-gram pairs use integer offsets from `-window` through
  `+window`, excluding zero.
- One root seed creates separate streams for initialization, sub-sampling,
  epoch ordering, and negative sampling.
- A negative draw equal to the current positive target is rejected and
  redrawn. Repeated negative draws otherwise remain valid.
- Input center vectors and output context vectors are separate matrices. Each
  step snapshots both, computes every gradient from old values, then writes
  both updates.
- All updated values, probabilities, and loss values are finite-checked.

## Run Locally

The environment already provides the required packages. Start the API:

```bash
python3 -m app.main
```

Default paths are `data/runs.sqlite` and `data/artifacts/`. Override them with
`SGNS_DB_PATH`, `SGNS_ARTIFACT_DIR`, `SGNS_HOST`, and `SGNS_PORT`.

Train a model:

```bash
curl -sS -X POST http://127.0.0.1:8000/train \
  -H 'Content-Type: application/json' \
  -d '{
    "corpus": [["river","stone","river"], ["forest","leaf","forest"]],
    "seed": 20260401,
    "epochs": 8,
    "window": 1,
    "negatives": 3,
    "dim": 8,
    "subsample_t": 1.0
  }'
```

Use the returned `run_id` with:

- `GET /health`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/events`
- `GET /vectors/{run_id}?token=river`
- `GET /similarity/{run_id}?token_a=river&token_b=stone`
- `GET /neighbors/{run_id}?token=river&k=2`
- `POST /evaluations/{run_id}` with `{"pairs":[["river","stone"]]}`

Run the local demonstration:

```bash
python3 scripts/demo.py
```

## Replay Evidence

Replay events contain the run ID, root seed, normalized config, SHA-256 corpus
fingerprint, frozen vocabulary, counts, frequencies, negative distribution,
RNG states, per-epoch pair order, per-step center/positive/negative IDs, loss,
scores/probabilities, gradient norms, and epoch result.

Run IDs are deterministic hashes of the corpus fingerprint and normalized
config, so a failed request is addressable and reproducible. Successful runs
are stored with their artifact; failed runs remain queryable but do not expose
an artifact.

Run tests and retain an output log:

```bash
pytest -q | tee testlogs/pytest-run.txt
```

`testlogs/evidence.jsonl` records test run IDs (`nodeid`), concrete expected
vs. actual values, judgment reasons, pass/fail outcome, duration, and failure
reason. Delete the file before a clean evidence run if desired.

## Error Semantics

The core and HTTP layer share these categories:

- `validation_error` -> HTTP 400 for malformed corpus/token/config, OOV query,
  invalid `k`, positive/negative collision, or malformed artifact.
- Pydantic request-shape errors remain FastAPI's default HTTP 422.
- `state_conflict` -> HTTP 409 for an unknown run ID, duplicate run ID, or
  requesting an artifact from a failed run.
- `resource_exhausted` -> HTTP 503 for rejected-negative budget exhaustion or
  an epoch with no trainable pairs after sub-sampling.
- `computation_failed` -> HTTP 500 for non-finite loss/gradients or unexpected
  kernel failure.

Error responses use:

```json
{"error": "category", "message": "what failed", "details": {}}
```

## Verification Coverage

- Hand-computed positive and negative logistic gradients use formulas in
  `tests/reference.py`, not the training implementation.
- Small dictionaries assert exact count/frequency and `count^0.75` values.
- Scripted RNG sequences assert sub-sampling draws, negative collision
  rejection, repeated negatives, and exhausted-budget classification.
- A fixed corpus trained twice asserts identical matrices, events, order, and
  loss.
- SQLite/artifact round trips assert exact recovery and OOV query failure.
- Repeated synthetic pairs assert top-rank co-occurrence recovery without
  claiming semantic evaluation.
- HTTP tests assert success and each required failure category.
