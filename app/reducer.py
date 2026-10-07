"""Reduction kernel: greedy 1-minimization under the block-removal
operator, with caching and checkpoint/resume."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, env_fingerprint
from .kernel import RunOutcome, cache_key, program_digest, run_candidate
from .models import Document, Verdict
from .store import Store


@dataclass
class ReductionResult:
    job_id: str
    status: str
    final_doc: Document
    initial_size: int
    final_size: int
    steps: int
    minimality_claim: str


class Reducer:
    """Removes one setting block at a time; a removal is kept only if the
    candidate still reproduces the TARGET error. Terminates when no single
    remaining removal preserves the target failure, i.e. the result is
    1-minimal under the block-removal operator only."""

    def __init__(self, settings: Settings, store: Store):
        self.settings = settings
        self.store = store
        self.prog_digest = program_digest(settings.target_program)
        self.env_fp = env_fingerprint(settings)

    def evaluate(self, doc: Document) -> RunOutcome:
        """Cached execution: key binds program digest + env + input."""
        key = cache_key(self.settings, self.prog_digest, doc)
        hit = self.store.cache_get(key)
        if hit is not None:
            return RunOutcome(
                Verdict(hit["verdict"]), f"cache hit: {hit['reason']}", None, ""
            )
        outcome = run_candidate(self.settings, doc)
        self.store.cache_put(
            key, self.prog_digest, self.env_fp, doc.digest(),
            outcome.verdict, outcome.reason,
        )
        return outcome

    def start(self, doc: Document) -> str:
        errors = doc.validate_syntax()
        if errors:
            raise ValueError(f"input violates declared syntax: {errors}")
        return self.store.create_job(self.prog_digest, doc)

    def run(self, job_id: str, max_steps: int | None = None) -> ReductionResult:
        """Run (or resume) a reduction job.

        State is checkpointed after every step, so a job stopped by
        `max_steps` (status 'interrupted') can be resumed by calling
        `run` again with the same job id.
        """
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(f"unknown job {job_id}")
        if job["status"] == "done":
            doc = Store.doc_from_canonical(job["current_doc"])
            return self._result(job_id, "done", doc, job["steps_done"], doc)

        doc = Store.doc_from_canonical(job["current_doc"])
        initial_doc = doc
        pending = job["pending_index"] or 0
        steps_done = job["steps_done"]
        budget = max_steps

        # Verify the (possibly resumed) candidate still fails as target.
        outcome = self.evaluate(doc)
        steps_done += 1
        self.store.log_step(job_id, steps_done, "verify", "-",
                            outcome.verdict, outcome.reason, doc.digest())
        if outcome.verdict != Verdict.TARGET:
            self.store.checkpoint(job_id, doc, pending, steps_done, "done")
            raise RuntimeError(
                f"job {job_id}: candidate does not reproduce the target "
                f"error ({outcome.verdict}: {outcome.reason}); refusing to reduce"
            )

        while True:
            removable = doc.removable_indexes()
            if pending >= len(removable):
                break  # no single removal preserves the failure -> 1-minimal
            if budget is not None and budget <= 0:
                self.store.checkpoint(job_id, doc, pending, steps_done,
                                      "interrupted")
                return self._result(job_id, "interrupted", doc, steps_done,
                                    initial_doc)
            budget = None if budget is None else budget - 1

            idx = removable[pending]
            block = doc.blocks[idx]
            candidate = doc.without(idx)
            outcome = self.evaluate(candidate)
            steps_done += 1
            if outcome.verdict == Verdict.TARGET:
                self.store.log_step(job_id, steps_done, "remove-accepted",
                                    block.masked_text(), outcome.verdict,
                                    outcome.reason, candidate.digest())
                doc = candidate
                pending = 0  # restart scan on the smaller candidate
            else:
                self.store.log_step(job_id, steps_done, "remove-rejected",
                                    block.masked_text(), outcome.verdict,
                                    outcome.reason, candidate.digest())
                pending += 1
            self.store.checkpoint(job_id, doc, pending, steps_done)

        self.store.checkpoint(job_id, doc, pending, steps_done, "done")
        return self._result(job_id, "done", doc, steps_done, initial_doc)

    def _result(self, job_id: str, status: str, doc: Document, steps: int,
                initial_doc: Document) -> ReductionResult:
        claim = (
            "1-minimal under the block-removal operator only "
            "(no claim of global minimum)"
            if status == "done"
            else "interrupted; no minimality claim"
        )
        return ReductionResult(
            job_id=job_id,
            status=status,
            final_doc=doc,
            initial_size=len(initial_doc.blocks),
            final_size=len(doc.blocks),
            steps=steps,
            minimality_claim=claim,
        )
