"""FastAPI service adapter for the local SGNS backend."""
from __future__ import annotations

import uuid

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from .config import TrainConfig
from .data import corpus_fingerprint, validate_corpus
from .errors import (
    AppError,
    ComputationError,
    ResourceExhaustedError,
    StateConflictError,
    ValidationError,
)
from .evaluation import evaluate_pairs
from .model import ModelArtifact
from .prediction import cosine_similarity, most_similar, vector_for
from .schemas import (
    EvaluationRequest,
    EvaluationResponse,
    EventsResponse,
    NeighborsResponse,
    RunResponse,
    SimilarityResponse,
    TrainRequest,
    TrainResponse,
    VectorResponse,
)
from .storage import ModelStore
from .training import make_run_id, train_skipgram


def create_app(store: ModelStore) -> FastAPI:
    app = FastAPI(
        title="Local Skip-Gram Negative Sampling",
        version="1.0.0",
        description=(
            "Deterministic small-corpus word-vector training with replay logs. "
            "Synthetic evaluations do not claim semantic understanding."
        ),
    )

    @app.exception_handler(AppError)
    async def app_error_handler(_request, exc: AppError):
        status = {
            ValidationError: 400,
            StateConflictError: 409,
            ResourceExhaustedError: 503,
            ComputationError: 500,
        }[type(exc)]
        return JSONResponse(status_code=status, content=exc.to_dict())

    @app.get("/health")
    async def health():
        return {"status": "ok", "database": store.db_path}

    @app.post("/train", response_model=TrainResponse, status_code=201)
    async def train(request: TrainRequest):
        corpus = validate_corpus(request.corpus)
        fields = request.model_dump()
        fields.pop("corpus")
        config = TrainConfig.from_dict(fields)
        run_id = make_run_id(corpus, config)
        fingerprint = corpus_fingerprint(corpus)
        if store.run_exists(run_id):
            raise StateConflictError(
                "identical corpus and training configuration already has a run",
                {"run_id": run_id},
            )
        events: list[dict] = []
        try:
            result = train_skipgram(
                corpus,
                config,
                run_id=run_id,
                event_listener=events.append,
            )
        except AppError as exc:
            if not store.run_exists(run_id):
                store.record_failure(
                    run_id,
                    config=config.to_dict(),
                    corpus=corpus,
                    fingerprint=fingerprint,
                    category=exc.category,
                    message=exc.message,
                    events=events,
                )
            raise
        except Exception as exc:  # defensive numeric/kernel boundary
            error = ComputationError(
                "training failed due to an unexpected computation error",
                {"exception": type(exc).__name__},
            )
            if not store.run_exists(run_id):
                store.record_failure(
                    run_id,
                    config=config.to_dict(),
                    corpus=corpus,
                    fingerprint=fingerprint,
                    category=error.category,
                    message=str(exc),
                    events=events,
                )
            raise error from exc

        store.save_training_run(result, corpus)
        return TrainResponse(
            run_id=result.run_id,
            status="completed",
            vocab_size=result.vocab.size,
            dim=result.artifact.dim,
            final_loss=result.final_loss,
            corpus_fingerprint=fingerprint,
        )

    def artifact_for(run_id: str) -> ModelArtifact:
        return store.load_artifact(run_id)

    @app.get("/runs/{run_id}", response_model=RunResponse)
    async def get_run(run_id: str):
        record = store.get_run(run_id)
        return RunResponse(
            run_id=record.run_id,
            status=record.status,
            config=record.config,
            corpus_fingerprint=record.corpus_fingerprint,
            final_loss=record.final_loss,
            error_category=record.error_category,
            error_message=record.error_message,
        )

    @app.get("/runs/{run_id}/events", response_model=EventsResponse)
    async def get_events(run_id: str):
        events = store.get_events(run_id)
        return EventsResponse(run_id=run_id, event_count=len(events), events=events)

    @app.get("/vectors/{run_id}", response_model=VectorResponse)
    async def get_vector(
        run_id: str,
        token: str = Query(..., min_length=1),
    ):
        artifact = artifact_for(run_id)
        return VectorResponse(
            run_id=run_id, token=token, vector=[float(x) for x in vector_for(artifact, token)]
        )

    @app.get("/similarity/{run_id}", response_model=SimilarityResponse)
    async def get_similarity(
        run_id: str,
        token_a: str = Query(..., min_length=1),
        token_b: str = Query(..., min_length=1),
    ):
        artifact = artifact_for(run_id)
        return SimilarityResponse(
            run_id=run_id,
            token_a=token_a,
            token_b=token_b,
            similarity=cosine_similarity(artifact, token_a, token_b),
        )

    @app.get("/neighbors/{run_id}", response_model=NeighborsResponse)
    async def get_neighbors(
        run_id: str,
        token: str = Query(..., min_length=1),
        k: int = Query(5, ge=1),
    ):
        artifact = artifact_for(run_id)
        return NeighborsResponse(
            run_id=run_id, token=token, neighbors=most_similar(artifact, token, k=k)
        )

    @app.post("/evaluations/{run_id}", response_model=EvaluationResponse, status_code=201)
    async def create_evaluation(run_id: str, request: EvaluationRequest):
        artifact = artifact_for(run_id)
        metrics = evaluate_pairs(artifact, request.pairs)
        evaluation_id = f"eval-{run_id}-{uuid.uuid4().hex[:8]}"
        store.save_evaluation(evaluation_id, run_id, metrics)
        return EvaluationResponse(
            evaluation_id=evaluation_id, run_id=run_id, metrics=metrics
        )

    return app
