"""Report interface: FastAPI app exposing job creation, status, step
evidence, and the final minimality report."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import Settings
from .models import Block, Document
from .reducer import Reducer
from .store import Store


class JobRequest(BaseModel):
    blocks: list[Block]
    max_steps: int | None = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    initial_size: int
    final_size: int
    steps: int
    minimality_claim: str


def create_app(settings: Settings | None = None,
               store: Store | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or Store(settings.db_path)
    reducer = Reducer(settings, store)

    app = FastAPI(title="delta-debugging-service")
    app.state.settings = settings
    app.state.store = store
    app.state.reducer = reducer

    @app.post("/jobs", response_model=JobResponse)
    def create_job(req: JobRequest) -> JobResponse:
        doc = Document(blocks=req.blocks)
        errors = doc.validate_syntax()
        if errors:
            raise HTTPException(
                422, detail={"reason": "syntax constraint violated",
                             "errors": errors})
        job_id = reducer.start(doc)
        try:
            result = reducer.run(job_id, max_steps=req.max_steps)
        except RuntimeError as exc:
            raise HTTPException(
                409, detail={"job_id": job_id, "reason": str(exc)})
        return JobResponse(
            job_id=result.job_id, status=result.status,
            initial_size=result.initial_size, final_size=result.final_size,
            steps=result.steps, minimality_claim=result.minimality_claim,
        )

    @app.post("/jobs/{job_id}/resume", response_model=JobResponse)
    def resume_job(job_id: str, max_steps: int | None = None) -> JobResponse:
        if store.get_job(job_id) is None:
            raise HTTPException(404, detail={"job_id": job_id,
                                             "reason": "unknown job"})
        try:
            result = reducer.run(job_id, max_steps=max_steps)
        except RuntimeError as exc:
            raise HTTPException(
                409, detail={"job_id": job_id, "reason": str(exc)})
        return JobResponse(
            job_id=result.job_id, status=result.status,
            initial_size=result.initial_size, final_size=result.final_size,
            steps=result.steps, minimality_claim=result.minimality_claim,
        )

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, detail={"job_id": job_id,
                                             "reason": "unknown job"})
        doc = Store.doc_from_canonical(job["current_doc"])
        return {
            "job_id": job_id,
            "status": job["status"],
            "program_digest": job["program_digest"],
            "steps_done": job["steps_done"],
            "current_size": len(doc.blocks),
            "current_doc_masked": doc.masked_summary(),
        }

    @app.get("/jobs/{job_id}/steps")
    def job_steps(job_id: str) -> list[dict]:
        if store.get_job(job_id) is None:
            raise HTTPException(404, detail={"job_id": job_id,
                                             "reason": "unknown job"})
        return store.get_steps(job_id)

    @app.get("/jobs/{job_id}/report")
    def job_report(job_id: str) -> dict:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, detail={"job_id": job_id,
                                             "reason": "unknown job"})
        doc = Store.doc_from_canonical(job["current_doc"])
        done = job["status"] == "done"
        return {
            "job_id": job_id,
            "status": job["status"],
            "final_doc_masked": doc.masked_summary(),
            "final_size": len(doc.blocks),
            "minimality_claim": (
                "1-minimal under the block-removal operator only "
                "(no claim of global minimum)"
                if done else "interrupted; no minimality claim"
            ),
            "evidence_steps": len(store.get_steps(job_id)),
        }

    return app


app = create_app()
