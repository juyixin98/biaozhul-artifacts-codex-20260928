"""Job orchestration: create, run, resume and report reduction jobs."""
from __future__ import annotations

import json
import os
import uuid

from .masking import mask_input
from .models import JobSpec, JobState, StructuredInput, Verdict
from .oracle import Oracle
from .reducer import DdminState, ddmin, verify_one_minimal
from .store import Store

MINIMALITY_SCOPE = (
    "1-minimal under the record-removal operator only: removing any single "
    "remaining record no longer deterministically reproduces the target "
    "error. This is not a claim of global minimality."
)


class JobNotFound(KeyError):
    pass


class JobService:
    def __init__(self, store: Store):
        self.store = store

    def create_job(self, spec: JobSpec, request_id: str) -> str:
        for part in spec.program:
            if part.endswith(".py") and not os.path.isfile(part):
                raise FileNotFoundError(f"program file not found: {part}")
        job_id = uuid.uuid4().hex[:12]
        oracle = Oracle(self.store, job_id, spec.program, spec.classifier, spec.env)
        self.store.create_job(
            job_id, spec.model_dump_json(), oracle.program_sha, oracle.env_sha
        )
        self.store.add_event(
            job_id, request_id, "info", "job_created",
            {
                "program": spec.program,
                "program_sha256": oracle.program_sha,
                "record_count": len(spec.input.records),
                # sensitive fields are masked before anything is persisted
                "input_records_masked": mask_input(spec.input.records, spec.sensitive_fields),
            },
        )
        return job_id

    def _load(self, job_id: str) -> tuple[dict, JobSpec]:
        job = self.store.get_job(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return job, JobSpec.model_validate_json(job["spec_json"])

    def run_job(self, job_id: str, request_id: str, max_live_queries: int = 0) -> dict:
        """Run or resume the reduction. A positive max_live_queries pauses
        the run (state=paused) after that many live program executions."""
        job, spec = self._load(job_id)
        if job["state"] == JobState.DONE.value:
            return self.get_report(job_id, request_id)
        if job["state"] == JobState.FAILED.value:
            return {"job_id": job_id, "state": "failed", "error": job["error"]}

        oracle = Oracle(self.store, job_id, spec.program, spec.classifier, spec.env)
        full_input = spec.input
        self.store.update_job(job_id, state=JobState.RUNNING.value)

        if job["checkpoint_json"] is None:
            initial = oracle.test(full_input)
            self.store.add_event(
                job_id, request_id, "info", "initial_check",
                {"verdict": initial.verdict.value, "reason": initial.reason},
            )
            if initial.verdict != Verdict.TARGET:
                error = (
                    "initial input does not reproduce the target error: "
                    f"verdict={initial.verdict.value} ({initial.reason})"
                )
                self.store.update_job(job_id, state=JobState.FAILED.value, error=error)
                self.store.add_event(job_id, request_id, "warning", "job_failed",
                                     {"reason": error})
                return {"job_id": job_id, "state": "failed", "error": error}
            state = DdminState(full_input.ids, 2)
        else:
            state = DdminState.from_json(json.loads(job["checkpoint_json"]))
            self.store.add_event(
                job_id, request_id, "info", "run_resumed",
                {"remaining_ids": state.current_ids, "granularity": state.granularity},
            )

        def should_stop() -> bool:
            return bool(max_live_queries) and oracle.live_executions >= max_live_queries

        def on_checkpoint(s: DdminState) -> None:
            self.store.update_job(job_id, checkpoint=s.to_json())

        def test(ids: list[str]) -> Verdict:
            return oracle.test(full_input.subset(ids)).verdict

        final_state, done = ddmin(
            state.current_ids, test, state=state,
            on_checkpoint=on_checkpoint, should_stop=should_stop,
        )
        self.store.update_job(job_id, checkpoint=final_state.to_json())

        if not done:
            self.store.update_job(job_id, state=JobState.PAUSED.value)
            self.store.add_event(
                job_id, request_id, "info", "run_paused",
                {
                    "reason": f"live query budget {max_live_queries} exhausted",
                    "remaining_ids": final_state.current_ids,
                    "granularity": final_state.granularity,
                    "live_executions": oracle.live_executions,
                },
            )
            return {"job_id": job_id, "state": "paused",
                    "remaining": len(final_state.current_ids)}

        ok, checks = verify_one_minimal(final_state.current_ids, test)
        minimized = full_input.subset(final_state.current_ids)
        result = {
            "job_id": job_id,
            "state": "done",
            "verdict": Verdict.TARGET.value,
            "original_size": len(full_input.records),
            "minimized_size": len(final_state.current_ids),
            "minimized_record_ids": final_state.current_ids,
            "minimized_input": json.loads(minimized.canonical_json()),
            "one_minimal": ok,
            "one_minimal_checks": checks,
            "minimality_scope": MINIMALITY_SCOPE,
            "live_executions": oracle.live_executions,
            "cache_hits": oracle.cache_hits,
        }
        self.store.update_job(job_id, state=JobState.DONE.value, result=result)
        self.store.add_event(
            job_id, request_id, "info", "run_finished",
            {
                "minimized_record_ids": final_state.current_ids,
                "one_minimal": ok,
                "live_executions": oracle.live_executions,
                "cache_hits": oracle.cache_hits,
            },
        )
        return result

    def get_report(self, job_id: str, request_id: str | None = None) -> dict:
        job, spec = self._load(job_id)
        base = {"job_id": job_id, "state": job["state"]}
        if job["state"] == JobState.DONE.value and job["result_json"]:
            return json.loads(job["result_json"])
        if job["state"] == JobState.FAILED.value:
            base["error"] = job["error"]
        if job["checkpoint_json"]:
            checkpoint = json.loads(job["checkpoint_json"])
            base["remaining_ids"] = checkpoint["current_ids"]
        return base

    def get_events(self, job_id: str) -> list[dict]:
        job = self.store.get_job(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return self.store.list_events(job_id)
