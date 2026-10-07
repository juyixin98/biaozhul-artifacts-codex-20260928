"""FastAPI report and control interface."""
from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException, Request

from .config import Settings
from .models import JobSpec
from .service import JobNotFound, JobService
from .store import Store


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = Store(settings.db_path)
    service = JobService(store)
    app = FastAPI(title="delta-debugging service", version="0.1.0")
    app.state.settings = settings
    app.state.store = store
    app.state.service = service

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.post("/jobs", status_code=201)
    def create_job(spec: JobSpec, request: Request):
        try:
            job_id = service.create_job(spec, request.state.request_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"job_id": job_id, "state": "pending",
                "request_id": request.state.request_id}

    @app.post("/jobs/{job_id}/run")
    def run_job(job_id: str, request: Request, max_live_queries: int = 0):
        """Run the reduction; resumes automatically when the job is paused.
        max_live_queries > 0 pauses the run after that many live executions."""
        try:
            return service.run_job(job_id, request.state.request_id, max_live_queries)
        except JobNotFound:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        try:
            return service.get_report(job_id)
        except JobNotFound:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")

    @app.get("/jobs/{job_id}/report")
    def job_report(job_id: str):
        try:
            return service.get_report(job_id)
        except JobNotFound:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")

    @app.get("/jobs/{job_id}/events")
    def job_events(job_id: str):
        try:
            return {"job_id": job_id, "events": service.get_events(job_id)}
        except JobNotFound:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")

    return app


app = create_app()
