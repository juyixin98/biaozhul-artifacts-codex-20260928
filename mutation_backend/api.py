"""FastAPI report and execution endpoints."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response, status

from . import __version__
from .config import get_settings
from .engine import run_mutation_analysis
from .models import MutationReport, MutationRequest
from .storage import EvidenceStore


app = FastAPI(
    title="Restricted Python Mutation Testing Backend",
    version=__version__,
)


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "engine_version": __version__, "database": str(settings.database_path)}


@app.post("/mutation-runs", response_model=MutationReport, status_code=status.HTTP_201_CREATED)
def create_mutation_run(request: MutationRequest, response: Response) -> MutationReport:
    settings = get_settings()
    work_root = settings.work_root
    work_root.mkdir(parents=True, exist_ok=True)
    report = run_mutation_analysis(request, work_root=work_root, request_id=request.request_id)
    EvidenceStore(settings.database_path).save_report(report)
    if request.request_id is not None:
        response.headers["X-Request-ID"] = request.request_id
    return report


@app.get("/mutation-runs", response_model=list[dict])
def list_mutation_runs(limit: int = 50) -> list[dict]:
    settings = get_settings()
    bounded_limit = min(max(limit, 1), 200)
    return EvidenceStore(settings.database_path).list_reports(bounded_limit)


@app.get("/mutation-runs/{request_id}", response_model=MutationReport)
def get_mutation_run(request_id: str) -> MutationReport:
    settings = get_settings()
    report = EvidenceStore(settings.database_path).get_report(request_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"no mutation run found for request {request_id}")
    return report
