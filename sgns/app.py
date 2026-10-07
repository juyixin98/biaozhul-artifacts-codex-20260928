"""FastAPI HTTP wiring for the skip-gram backend."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .errors import SGNSError
from .schema import (
    CorpusIn,
    CorpusOut,
    EvaluateIn,
    EvaluateOut,
    NeighboursIn,
    NeighboursOut,
    RunSummary,
    SimilarityIn,
    SimilarityOut,
    TrainIn,
    TrainOut,
    VectorOut,
)
from .service import EmbeddingService


def create_app(
    db_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> FastAPI:
    db_path = db_path or os.environ.get(
        "SGNS_DB_PATH",
        str(Path.cwd() / "var" / "sgns.db"),
    )
    artifact_dir = artifact_dir or os.environ.get(
        "SGNS_ARTIFACT_DIR",
        str(Path.cwd() / "var" / "artifacts"),
    )
    service = EmbeddingService(db_path=db_path, artifact_dir=artifact_dir)
    app = FastAPI(
        title="Small Skip-gram Negative Sampling Backend",
        version="1.0.0",
    )
    app.state.service = service

    @app.exception_handler(SGNSError)
    async def handle_sgns_error(
        request: Request,
        exc: SGNSError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_dict(),
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/corpora", response_model=CorpusOut, status_code=201)
    async def create_corpus(payload: CorpusIn) -> CorpusOut:
        result = service.create_corpus(
            payload.corpus_id,
            payload.text,
            payload.max_tokens,
        )
        return CorpusOut(**result)

    @app.post("/training-runs", response_model=TrainOut, status_code=201)
    async def start_run(payload: TrainIn) -> TrainOut:
        result = service.start_run(
            run_id=payload.run_id,
            corpus_id=payload.corpus_id,
            config_overrides=payload.config,
            record_events=payload.record_events,
        )
        return TrainOut(**result)

    @app.get("/training-runs/{run_id}", response_model=RunSummary)
    async def get_run(run_id: str) -> RunSummary:
        return RunSummary(**service.get_run(run_id))

    @app.get("/training-runs/{run_id}/vectors/{word}", response_model=VectorOut)
    async def get_vector(run_id: str, word: str) -> VectorOut:
        return VectorOut(**service.vector(run_id, word))

    @app.post(
        "/training-runs/{run_id}/similarity",
        response_model=SimilarityOut,
    )
    async def similarity(run_id: str, payload: SimilarityIn) -> SimilarityOut:
        return SimilarityOut(
            **service.similarity(run_id, payload.word_a, payload.word_b)
        )

    @app.post(
        "/training-runs/{run_id}/neighbours",
        response_model=NeighboursOut,
    )
    async def neighbours(run_id: str, payload: NeighboursIn) -> NeighboursOut:
        return NeighboursOut(
            **service.neighbours(run_id, payload.word, payload.top_k)
        )

    @app.post(
        "/training-runs/{run_id}/evaluate",
        response_model=EvaluateOut,
    )
    async def evaluate(run_id: str, payload: EvaluateIn) -> EvaluateOut:
        pairs = [pair.model_dump() for pair in payload.pairs]
        return EvaluateOut(**service.evaluate(run_id, pairs))

    return app


app = create_app()
