"""FastAPI reporting + control interface.

Endpoints
---------
GET  /healthz
POST /models/validate              validate a model document (input errors)
POST /runs                         start a run against a local or HTTP SUT
GET  /runs                         list runs
GET  /runs/{run_id}                run header + summary + error contract
GET  /runs/{run_id}/steps          per-step evidence (states, rationale)
GET  /runs/{run_id}/findings       discrepancies + reduction metadata
GET  /runs/{run_id}/findings/{i}/replay   minimal replay script text
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from . import campaign as campaign_mod
from .campaign import run_campaign
from .errors import ErrorCategory, InputError, MdbtError
from .executor import HttpTransport, InProcessTransport
from .services.buggy import create_app as create_buggy_app
from .services.reference import create_app as create_reference_app
from .spec import load_spec
from .storage import EvidenceStore
from .services.common import (
    unhandled_exception_handler,
    validation_exception_handler as mdbt_validation_handler,
)


LOCAL_SUITES = {
    "reference": (
        "reference-order-service",
        create_reference_app,
    ),
    "buggy": (
        "buggy-order-service",
        create_buggy_app,
    ),
}


class StartRunRequest(BaseModel):
    sut: Literal["reference", "buggy", "http"] = "buggy"
    base_url: str | None = Field(default=None, description="required when sut='http'")
    model: dict[str, Any]
    seed: int = 20261007
    fuzz_steps: int = Field(default=40, ge=0, le=2000)
    illegal_fraction: float = Field(default=0.45, ge=0.0, le=1.0)
    db_path: str = "artifacts/evidence.db"
    replay_dir: str = "artifacts/replays"


class ValidateRequest(BaseModel):
    model: dict[str, Any]


def create_app() -> FastAPI:
    app = FastAPI(title="model-driven-backend-tester", version="1.0.0")
    app.add_exception_handler(RequestValidationError, mdbt_validation_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    def _store(req: StartRunRequest | None = None, db_path: str | None = None):
        return EvidenceStore(db_path or (req.db_path if req else "artifacts/evidence.db"))

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "service": "mdbt-api", "suites": sorted(LOCAL_SUITES)}

    @app.post("/models/validate")
    def validate_model(req: ValidateRequest) -> dict:
        try:
            spec = load_spec(req.model)
        except InputError as exc:
            return {"valid": False, "error": exc.to_dict()}
        return {"valid": True, "model": spec.to_dict()}

    @app.post("/runs")
    async def start_run(req: StartRunRequest) -> dict:
        try:
            spec = load_spec(req.model)
        except InputError as exc:
            raise HTTPException(status_code=422, detail=exc.to_dict())

        if req.sut == "http":
            if not req.base_url:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "category": ErrorCategory.INPUT_ERROR.value,
                        "message": "base_url is required when sut='http'",
                    },
                )

            base_url = req.base_url

            def transport_factory():
                return HttpTransport(base_url)

            sut_name = f"http:{req.base_url}"
            sut_target = req.base_url
        else:
            sut_name, app_factory_local = LOCAL_SUITES[req.sut]

            def transport_factory():
                return InProcessTransport(app_factory_local())

            sut_target = "in-process"

        with _store(req) as store:
            run_id = store.new_run_id()
            result = await run_campaign(
                spec=spec,
                sut_name=sut_name,
                transport_factory=transport_factory,
                store=store,
                run_id=run_id,
                seed=req.seed,
                fuzz_steps=req.fuzz_steps,
                illegal_fraction=req.illegal_fraction,
                replay_dir=req.replay_dir,
                sut_target=sut_target,
            )
        return result

    @app.get("/runs")
    def list_runs(db_path: str = "artifacts/evidence.db", limit: int = 50) -> dict:
        with _store(db_path=db_path) as store:
            return {"runs": store.list_runs(limit=limit)}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, db_path: str = "artifacts/evidence.db") -> dict:
        with _store(db_path=db_path) as store:
            run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail={"run_id": run_id})
        return run

    @app.get("/runs/{run_id}/steps")
    def get_steps(run_id: str, db_path: str = "artifacts/evidence.db") -> dict:
        with _store(db_path=db_path) as store:
            if store.get_run(run_id) is None:
                raise HTTPException(status_code=404, detail={"run_id": run_id})
            return {"run_id": run_id, "steps": store.get_steps(run_id)}

    @app.get("/runs/{run_id}/findings")
    def get_findings(run_id: str, db_path: str = "artifacts/evidence.db") -> dict:
        with _store(db_path=db_path) as store:
            if store.get_run(run_id) is None:
                raise HTTPException(status_code=404, detail={"run_id": run_id})
            findings = store.get_findings(run_id)
        for f in findings:
            f.pop("replay_script", None)
        return {"run_id": run_id, "findings": findings}

    @app.get("/runs/{run_id}/findings/{index}/replay")
    def get_replay(
        run_id: str, index: int, db_path: str = "artifacts/evidence.db"
    ):
        from fastapi.responses import PlainTextResponse

        with _store(db_path=db_path) as store:
            findings = store.get_findings(run_id)
        if not findings or not (0 <= index < len(findings)):
            raise HTTPException(
                status_code=404,
                detail={"run_id": run_id, "finding_index": index},
            )
        return PlainTextResponse(findings[index]["replay_script"])

    return app


app = create_app()
