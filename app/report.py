"""FastAPI report interface for the mutation testing backend."""
from __future__ import annotations

import logging
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from . import __version__
from .config import get_settings
from .models import MutantRecord, RunRequest, RunSummary
from .runner import MutationRunner
from .store import EvidenceStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s "
    "[run=%(run_id)s requester=%(requester)s] %(message)s",
    defaults={"run_id": "-", "requester": "-"},
)
logger = logging.getLogger("mutation.api")

settings = get_settings()
store = EvidenceStore(settings.db_path)
runner = MutationRunner(settings, store)

app = FastAPI(title="Restricted Mutation Testing Backend",
              version=__version__)


@app.middleware("http")
async def request_identity(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "backend_version": __version__,
        "operator_set_version": settings.operator_set_version,
    }


@app.post("/runs", response_model=RunSummary, status_code=202)
def start_run(req: RunRequest, request: Request,
              background: BackgroundTasks) -> RunSummary:
    log = logging.LoggerAdapter(
        logger, {"run_id": "pending",
                 "requester": f"{req.requester}/{request.state.request_id}"}
    )
    log.info("accepted mutation run for %s", req.source_dir)
    background.add_task(runner.run, req)
    # Return a placeholder summary; the stored run is the durable record.
    return RunSummary(
        run_id="pending",
        status="completed",  # type: ignore[arg-type]
        requester=req.requester,
        source_dir=req.source_dir,
        source_digest="",
        operator_set_version=settings.operator_set_version,
        python_version="",
        created_at="",
        baseline_note="run scheduled; poll GET /runs for the stored record",
    )


@app.post("/runs/sync", response_model=RunSummary)
def run_sync(req: RunRequest, request: Request) -> RunSummary:
    """Synchronous run: convenient for fixtures and verification."""
    from pathlib import Path
    if not Path(req.source_dir).is_dir():
        raise HTTPException(status_code=400, detail="source_dir not found")
    logging.LoggerAdapter(
        logger, {"run_id": "sync",
                 "requester": f"{req.requester}/{request.state.request_id}"}
    ).info("synchronous run requested for %s", req.source_dir)
    return runner.run(req)


@app.get("/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    return store.list_runs()


@app.get("/runs/{run_id}", response_model=RunSummary)
def get_run(run_id: str) -> RunSummary:
    summary = store.get_run(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="run not found")
    return summary


@app.get("/runs/{run_id}/mutants", response_model=list[MutantRecord])
def get_mutants(run_id: str) -> list[MutantRecord]:
    if store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return store.list_mutants(run_id)


@app.get("/runs/{run_id}/report")
def get_report(run_id: str) -> dict:
    """Human-oriented report: failures and undetermined mutants separated."""
    summary = store.get_run(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="run not found")
    mutants = store.list_mutants(run_id)
    failures = [
        m for m in mutants
        if m.outcome.value in ("invalid_syntax", "startup_failure", "timeout")
    ]
    undetermined = [m for m in mutants if m.outcome.value == "undetermined"]
    survived = [m for m in mutants if m.outcome.value == "survived"]
    return {
        "summary": summary.model_dump(),
        "survived_mutants": [m.model_dump() for m in survived],
        "failures": [m.model_dump() for m in failures],
        "undetermined": [m.model_dump() for m in undetermined],
    }
